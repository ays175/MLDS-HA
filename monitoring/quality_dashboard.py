#!/usr/bin/env python3
"""
Simple Quality Monitoring Dashboard
Provides real-time view of processing pipeline health
"""
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.global_config import get_global_config
from ingestion.batch_processor import BatchProcessor


class QualityDashboard:
    """Simple quality monitoring dashboard for MLDS pipeline"""
    
    def __init__(self):
        self.config = get_global_config()
        self.batch_processor = BatchProcessor()
        self.platforms = ['facebook', 'instagram', 'tiktok', 'customer_care']
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        try:
            self.batch_processor.connect_weaviate()
            
            # Get processing status for all platforms
            platform_status = self.batch_processor.get_processing_status()
            
            # Calculate overall health metrics
            total_records = sum(
                status.get('total_records', 0) 
                for status in platform_status.values() 
                if isinstance(status, dict)
            )
            
            # Average coverage across platforms
            sentiment_coverages = [
                status.get('sentiment_percentage', 0) 
                for status in platform_status.values() 
                if isinstance(status, dict) and status.get('total_records', 0) > 0
            ]
            
            semantic_coverages = [
                status.get('semantic_percentage', 0) 
                for status in platform_status.values() 
                if isinstance(status, dict) and status.get('total_records', 0) > 0
            ]
            
            avg_sentiment_coverage = sum(sentiment_coverages) / len(sentiment_coverages) if sentiment_coverages else 0
            avg_semantic_coverage = sum(semantic_coverages) / len(semantic_coverages) if semantic_coverages else 0
            
            # Determine overall health status
            min_sentiment = self.config.get('monitoring.min_sentiment_coverage', 80)
            min_semantic = self.config.get('monitoring.min_semantic_coverage', 50)
            
            if avg_sentiment_coverage >= min_sentiment and avg_semantic_coverage >= min_semantic:
                health_status = "healthy"
            elif avg_sentiment_coverage >= min_sentiment * 0.7:
                health_status = "warning"
            else:
                health_status = "critical"
            
            return {
                "overall_status": health_status,
                "total_records": total_records,
                "avg_sentiment_coverage": round(avg_sentiment_coverage, 1),
                "avg_semantic_coverage": round(avg_semantic_coverage, 1),
                "platform_count": len([p for p in platform_status.values() if isinstance(p, dict)]),
                "platforms": platform_status,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "overall_status": "error",
                "error": str(e),
                "last_updated": datetime.now().isoformat()
            }
    
    def get_processing_pipeline_status(self) -> Dict[str, Any]:
        """Get detailed processing pipeline status"""
        try:
            self.batch_processor.connect_weaviate()
            platform_status = self.batch_processor.get_processing_status()
            
            pipeline_status = {}
            
            for platform in self.platforms:
                status = platform_status.get(platform, {})
                
                if isinstance(status, dict):
                    total = status.get('total_records', 0)
                    status_counts = status.get('status_counts', {})
                    
                    # Calculate pipeline stage completion
                    ingested = status_counts.get('ingested', 0)
                    sentiment_complete = status_counts.get('sentiment_complete', 0)
                    semantic_complete = status_counts.get('semantic_complete', 0)
                    fully_processed = status_counts.get('fully_processed', 0)
                    
                    pipeline_status[platform] = {
                        "total_records": total,
                        "pipeline_stages": {
                            "1_ingested": {"count": ingested, "percentage": round(ingested/max(1,total)*100, 1)},
                            "2_sentiment": {"count": sentiment_complete + semantic_complete + fully_processed, 
                                          "percentage": round((sentiment_complete + semantic_complete + fully_processed)/max(1,total)*100, 1)},
                            "3_semantic": {"count": semantic_complete + fully_processed, 
                                         "percentage": round((semantic_complete + fully_processed)/max(1,total)*100, 1)},
                            "4_complete": {"count": fully_processed, "percentage": round(fully_processed/max(1,total)*100, 1)}
                        },
                        "bottlenecks": self._identify_bottlenecks(status_counts, total)
                    }
                else:
                    pipeline_status[platform] = {"error": status}
            
            return {
                "pipeline_status": pipeline_status,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _identify_bottlenecks(self, status_counts: Dict[str, int], total: int) -> List[str]:
        """Identify processing bottlenecks"""
        bottlenecks = []
        
        ingested = status_counts.get('ingested', 0)
        sentiment_complete = status_counts.get('sentiment_complete', 0)
        semantic_complete = status_counts.get('semantic_complete', 0)
        
        # Check for bottlenecks
        if ingested > total * 0.3:  # More than 30% stuck at ingestion
            bottlenecks.append("sentiment_processing")
        
        if sentiment_complete > total * 0.2:  # More than 20% stuck at sentiment
            bottlenecks.append("semantic_processing")
        
        if total > 0 and (sentiment_complete + semantic_complete) / total > 0.8:
            bottlenecks.append("final_processing")
        
        return bottlenecks
    
    def get_recent_job_performance(self, days: int = 7) -> Dict[str, Any]:
        """Get recent job performance metrics"""
        try:
            # Load job history from scheduler
            job_files = list(Path("logs").glob("job_history_*.json"))
            
            if not job_files:
                return {"error": "No job history found"}
            
            # Load most recent job history
            latest_file = max(job_files, key=lambda x: x.stat().st_mtime)
            
            with open(latest_file, 'r') as f:
                job_history = json.load(f)
            
            # Filter to recent jobs
            cutoff = datetime.now() - timedelta(days=days)
            recent_jobs = [
                job for job in job_history 
                if datetime.fromisoformat(job['start_time']) > cutoff
            ]
            
            if not recent_jobs:
                return {"message": f"No jobs found in last {days} days"}
            
            # Analyze job performance
            total_jobs = len(recent_jobs)
            successful_jobs = len([j for j in recent_jobs if j['status'] == 'success'])
            failed_jobs = len([j for j in recent_jobs if j['status'] == 'error'])
            
            # Average duration by job type
            job_durations = {}
            for job in recent_jobs:
                job_type = job['job_type']
                duration = job.get('duration_minutes', 0)
                
                if job_type not in job_durations:
                    job_durations[job_type] = []
                job_durations[job_type].append(duration)
            
            avg_durations = {
                job_type: round(sum(durations) / len(durations), 1)
                for job_type, durations in job_durations.items()
            }
            
            return {
                "period_days": days,
                "total_jobs": total_jobs,
                "success_rate": round(successful_jobs / total_jobs * 100, 1),
                "failed_jobs": failed_jobs,
                "avg_durations_minutes": avg_durations,
                "recent_failures": [
                    {"job_id": j["job_id"], "job_type": j["job_type"], "error": j.get("error", "")}
                    for j in recent_jobs if j["status"] == "error"
                ][-5:]  # Last 5 failures
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def generate_dashboard_report(self) -> Dict[str, Any]:
        """Generate comprehensive dashboard report"""
        return {
            "dashboard_generated_at": datetime.now().isoformat(),
            "system_health": self.get_system_health(),
            "processing_pipeline": self.get_processing_pipeline_status(),
            "job_performance": self.get_recent_job_performance(),
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on current status"""
        recommendations = []
        
        try:
            health = self.get_system_health()
            
            # Check sentiment coverage
            if health.get('avg_sentiment_coverage', 0) < 80:
                recommendations.append("Run sentiment batch processing to improve coverage")
            
            # Check semantic coverage
            if health.get('avg_semantic_coverage', 0) < 50:
                recommendations.append("Schedule semantic analysis for platforms with sufficient data")
            
            # Check job performance
            job_perf = self.get_recent_job_performance()
            if job_perf.get('success_rate', 100) < 90:
                recommendations.append("Investigate recent job failures and improve error handling")
            
            # Platform-specific recommendations
            platforms = health.get('platforms', {})
            for platform, status in platforms.items():
                if isinstance(status, dict):
                    sentiment_pct = status.get('sentiment_percentage', 0)
                    if sentiment_pct < 50:
                        recommendations.append(f"Priority: Process pending sentiment for {platform}")
            
            if not recommendations:
                recommendations.append("System is healthy - continue regular monitoring")
            
        except Exception as e:
            recommendations.append(f"Error generating recommendations: {e}")
        
        return recommendations
    
    def print_dashboard(self):
        """Print dashboard to console"""
        report = self.generate_dashboard_report()
        
        print("=" * 60)
        print("🏥 MLDS QUALITY MONITORING DASHBOARD")
        print("=" * 60)
        
        # System Health
        health = report['system_health']
        status_emoji = {"healthy": "✅", "warning": "⚠️", "critical": "❌", "error": "💥"}
        
        print(f"\n📊 SYSTEM HEALTH: {status_emoji.get(health['overall_status'], '❓')} {health['overall_status'].upper()}")
        print(f"   Total Records: {health.get('total_records', 0):,}")
        print(f"   Sentiment Coverage: {health.get('avg_sentiment_coverage', 0)}%")
        print(f"   Semantic Coverage: {health.get('avg_semantic_coverage', 0)}%")
        
        # Platform Status
        print(f"\n🔍 PLATFORM STATUS:")
        platforms = health.get('platforms', {})
        for platform, status in platforms.items():
            if isinstance(status, dict):
                total = status.get('total_records', 0)
                sentiment_pct = status.get('sentiment_percentage', 0)
                semantic_pct = status.get('semantic_percentage', 0)
                print(f"   {platform:12}: {total:6,} records | Sentiment: {sentiment_pct:5.1f}% | Semantic: {semantic_pct:5.1f}%")
        
        # Job Performance
        job_perf = report['job_performance']
        if 'success_rate' in job_perf:
            print(f"\n⚡ JOB PERFORMANCE (Last 7 days):")
            print(f"   Success Rate: {job_perf['success_rate']}%")
            print(f"   Total Jobs: {job_perf['total_jobs']}")
            print(f"   Failed Jobs: {job_perf['failed_jobs']}")
        
        # Recommendations
        recommendations = report['recommendations']
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations[:5], 1):
            print(f"   {i}. {rec}")
        
        print(f"\n🕒 Last Updated: {health.get('last_updated', 'Unknown')}")
        print("=" * 60)


def main():
    """CLI interface for quality dashboard"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MLDS Quality Monitoring Dashboard")
    parser.add_argument("--format", choices=['console', 'json'], default='console',
                       help="Output format")
    parser.add_argument("--output", help="Output file (for JSON format)")
    
    args = parser.parse_args()
    
    dashboard = QualityDashboard()
    
    if args.format == 'console':
        dashboard.print_dashboard()
    else:
        report = dashboard.generate_dashboard_report()
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"Dashboard report saved to: {args.output}")
        else:
            print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
