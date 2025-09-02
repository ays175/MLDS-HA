#!/usr/bin/env python3
"""
Simple Scheduler for Batch Processing Jobs
Manages scheduled sentiment analysis, semantic analysis, and metrics generation
"""
import time
import json
import schedule
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.global_config import get_global_config
from ingestion.batch_processor import BatchProcessor


class MLDSScheduler:
    """Simple scheduler for MLDS batch processing jobs"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = get_global_config()
        self.batch_processor = BatchProcessor()
        self.platforms = ['facebook', 'instagram', 'tiktok', 'customer_care']
        self.job_history = []
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging for scheduler"""
        log_level = self.config.get('logging.level', 'INFO')
        log_format = self.config.get('logging.format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('logs/scheduler.log', mode='a')
            ]
        )
        
        self.logger = logging.getLogger('MLDSScheduler')
    
    def setup_default_jobs(self):
        """Setup default batch processing jobs"""
        self.logger.info("Setting up default batch processing jobs...")
        
        # Daily sentiment processing (staggered by platform)
        schedule.every().day.at("02:00").do(self._run_sentiment_job, "facebook")
        schedule.every().day.at("02:30").do(self._run_sentiment_job, "instagram") 
        schedule.every().day.at("03:00").do(self._run_sentiment_job, "tiktok")
        schedule.every().day.at("03:30").do(self._run_sentiment_job, "customer_care")
        
        # Weekly semantic analysis (Sunday early morning)
        schedule.every().sunday.at("01:00").do(self._run_semantic_job, "all")
        
        # Daily metrics generation (after sentiment processing)
        schedule.every().day.at("05:00").do(self._run_metrics_job, "all")
        
        # Weekly system health check
        schedule.every().sunday.at("06:00").do(self._run_health_check)
        
        self.logger.info("✅ Default jobs scheduled")
        self._log_next_jobs()
    
    def setup_custom_jobs(self, jobs_config: Dict[str, Any]):
        """Setup custom jobs from configuration"""
        self.logger.info("Setting up custom batch processing jobs...")
        
        for job_name, job_config in jobs_config.items():
            job_type = job_config.get('type')
            schedule_str = job_config.get('schedule')
            platforms = job_config.get('platforms', ['all'])
            
            if job_type == 'sentiment':
                for platform in platforms:
                    schedule.every().day.at(schedule_str).do(self._run_sentiment_job, platform)
            elif job_type == 'semantic':
                schedule.every().week.at(schedule_str).do(self._run_semantic_job, platforms[0])
            elif job_type == 'metrics':
                schedule.every().day.at(schedule_str).do(self._run_metrics_job, platforms[0])
        
        self.logger.info(f"✅ {len(jobs_config)} custom jobs scheduled")
    
    def _run_sentiment_job(self, platform: str) -> Dict[str, Any]:
        """Run sentiment analysis job for a platform"""
        job_id = f"sentiment_{platform}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"🧠 Starting sentiment job: {job_id}")
        start_time = datetime.now()
        
        try:
            # Connect to Weaviate
            self.batch_processor.connect_weaviate()
            
            # Process sentiment
            result = self.batch_processor.process_pending_sentiment(platform)
            
            # Record job completion
            job_record = {
                "job_id": job_id,
                "job_type": "sentiment",
                "platform": platform,
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_minutes": (datetime.now() - start_time).total_seconds() / 60,
                "status": "success" if result.get("processed", 0) > 0 else "no_data",
                "records_processed": result.get("processed", 0),
                "result": result
            }
            
            self.job_history.append(job_record)
            self.logger.info(f"✅ Sentiment job completed: {result.get('processed', 0)} records processed")
            
            return job_record
            
        except Exception as e:
            error_record = {
                "job_id": job_id,
                "job_type": "sentiment", 
                "platform": platform,
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "status": "error",
                "error": str(e)
            }
            
            self.job_history.append(error_record)
            self.logger.error(f"❌ Sentiment job failed: {e}")
            
            return error_record
    
    def _run_semantic_job(self, platform: str) -> Dict[str, Any]:
        """Run semantic analysis job"""
        job_id = f"semantic_{platform}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"🔬 Starting semantic job: {job_id}")
        start_time = datetime.now()
        
        try:
            # Connect to Weaviate
            self.batch_processor.connect_weaviate()
            
            if platform == "all":
                # Run for all platforms
                results = {}
                for plat in self.platforms:
                    results[plat] = self.batch_processor.process_semantic_analysis(plat)
            else:
                # Run for specific platform
                results = {platform: self.batch_processor.process_semantic_analysis(platform)}
            
            # Record job completion
            total_topics = sum(r.get("topics_discovered", 0) for r in results.values() if isinstance(r, dict))
            
            job_record = {
                "job_id": job_id,
                "job_type": "semantic",
                "platform": platform,
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_minutes": (datetime.now() - start_time).total_seconds() / 60,
                "status": "success",
                "topics_discovered": total_topics,
                "results": results
            }
            
            self.job_history.append(job_record)
            self.logger.info(f"✅ Semantic job completed: {total_topics} topics discovered")
            
            return job_record
            
        except Exception as e:
            error_record = {
                "job_id": job_id,
                "job_type": "semantic",
                "platform": platform,
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "status": "error",
                "error": str(e)
            }
            
            self.job_history.append(error_record)
            self.logger.error(f"❌ Semantic job failed: {e}")
            
            return error_record
    
    def _run_metrics_job(self, platform: str) -> Dict[str, Any]:
        """Run metrics generation job"""
        job_id = f"metrics_{platform}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"📊 Starting metrics job: {job_id}")
        start_time = datetime.now()
        
        try:
            # Import metrics exporter
            from metrics.unified_metrics_export import UnifiedMetricsExporter
            
            exporter = UnifiedMetricsExporter()
            exporter.connect_weaviate()
            
            if platform == "all":
                # Export metrics for all platforms
                platforms = ['facebook', 'instagram', 'tiktok', 'customer_care']
                results = {}
                for plat in platforms:
                    try:
                        metrics = exporter.export_platform_metrics(plat)
                        results[plat] = {"status": "success", "records": metrics.get("total_records", 0)}
                    except Exception as e:
                        results[plat] = {"status": "error", "error": str(e)}
            else:
                # Export for specific platform
                metrics = exporter.export_platform_metrics(platform)
                results = {"status": "success", "records": metrics.get("total_records", 0)}
            
            job_record = {
                "job_id": job_id,
                "job_type": "metrics",
                "platform": platform,
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_minutes": (datetime.now() - start_time).total_seconds() / 60,
                "status": "success",
                "results": results
            }
            
            self.job_history.append(job_record)
            self.logger.info(f"✅ Metrics job completed")
            
            return job_record
            
        except Exception as e:
            error_record = {
                "job_id": job_id,
                "job_type": "metrics",
                "platform": platform,
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "status": "error",
                "error": str(e)
            }
            
            self.job_history.append(error_record)
            self.logger.error(f"❌ Metrics job failed: {e}")
            
            return error_record
    
    def _run_health_check(self) -> Dict[str, Any]:
        """Run system health check"""
        job_id = f"health_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"🏥 Starting health check: {job_id}")
        start_time = datetime.now()
        
        try:
            # Connect to Weaviate
            self.batch_processor.connect_weaviate()
            
            # Get processing status for all platforms
            status = self.batch_processor.get_processing_status()
            
            # Check for issues using configurable thresholds
            min_sentiment_coverage = self.config.get('monitoring.min_sentiment_coverage', 80)
            min_semantic_coverage = self.config.get('monitoring.min_semantic_coverage', 50)
            max_pending_records = self.config.get('monitoring.max_pending_records', 10000)
            
            issues = []
            for platform, plat_status in status.items():
                if isinstance(plat_status, dict):
                    sentiment_pct = plat_status.get('sentiment_percentage', 0)
                    semantic_pct = plat_status.get('semantic_percentage', 0)
                    total_records = plat_status.get('total_records', 0)
                    
                    # Check sentiment coverage
                    if sentiment_pct < min_sentiment_coverage:
                        issues.append(f"{platform}: Only {sentiment_pct}% sentiment coverage (target: {min_sentiment_coverage}%)")
                    
                    # Check semantic coverage
                    if semantic_pct < min_semantic_coverage and total_records > 1000:
                        issues.append(f"{platform}: Only {semantic_pct}% semantic coverage (target: {min_semantic_coverage}%)")
                    
                    # Check for too many pending records
                    pending_sentiment = total_records * (100 - sentiment_pct) / 100
                    if pending_sentiment > max_pending_records:
                        issues.append(f"{platform}: {int(pending_sentiment)} records pending sentiment (max: {max_pending_records})")
            
            job_record = {
                "job_id": job_id,
                "job_type": "health_check",
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "status": "success",
                "platform_status": status,
                "issues_found": len(issues),
                "issues": issues
            }
            
            self.job_history.append(job_record)
            
            if issues:
                self.logger.warning(f"⚠️ Health check found {len(issues)} issues: {issues}")
            else:
                self.logger.info("✅ Health check passed - all systems healthy")
            
            return job_record
            
        except Exception as e:
            error_record = {
                "job_id": job_id,
                "job_type": "health_check",
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "status": "error",
                "error": str(e)
            }
            
            self.job_history.append(error_record)
            self.logger.error(f"❌ Health check failed: {e}")
            
            return error_record
    
    def _log_next_jobs(self):
        """Log information about next scheduled jobs"""
        jobs = schedule.get_jobs()
        if jobs:
            self.logger.info(f"📅 Next {min(5, len(jobs))} scheduled jobs:")
            for i, job in enumerate(sorted(jobs, key=lambda x: x.next_run)[:5]):
                next_run = job.next_run.strftime("%Y-%m-%d %H:%M:%S")
                self.logger.info(f"  {i+1}. {job.job_func.__name__} at {next_run}")
    
    def save_job_history(self, output_file: str = None):
        """Save job history to file"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d")
            output_file = f"logs/job_history_{timestamp}.json"
        
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(self.job_history, f, indent=2, default=str)
        
        self.logger.info(f"💾 Job history saved to: {output_file}")
    
    def get_job_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get summary of jobs from last N days"""
        cutoff = datetime.now() - timedelta(days=days)
        
        recent_jobs = [
            job for job in self.job_history 
            if datetime.fromisoformat(job['start_time']) > cutoff
        ]
        
        summary = {
            "period_days": days,
            "total_jobs": len(recent_jobs),
            "successful_jobs": len([j for j in recent_jobs if j['status'] == 'success']),
            "failed_jobs": len([j for j in recent_jobs if j['status'] == 'error']),
            "job_types": {},
            "platforms": {},
            "avg_duration_minutes": 0
        }
        
        # Job type breakdown
        for job in recent_jobs:
            job_type = job['job_type']
            summary['job_types'][job_type] = summary['job_types'].get(job_type, 0) + 1
        
        # Platform breakdown
        for job in recent_jobs:
            platform = job.get('platform', 'unknown')
            summary['platforms'][platform] = summary['platforms'].get(platform, 0) + 1
        
        # Average duration
        durations = [job.get('duration_minutes', 0) for job in recent_jobs if job.get('duration_minutes')]
        if durations:
            summary['avg_duration_minutes'] = sum(durations) / len(durations)
        
        return summary
    
    def run(self):
        """Run the scheduler (blocking)"""
        self.logger.info("🚀 MLDS Scheduler starting...")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            self.logger.info("⏹️ Scheduler stopped by user")
        except Exception as e:
            self.logger.error(f"💥 Scheduler crashed: {e}")
            raise
        finally:
            # Save job history on exit
            self.save_job_history()
            
            # Close batch processor connection
            if self.batch_processor.client:
                self.batch_processor.client.close()


def main():
    """CLI interface for scheduler"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MLDS Batch Processing Scheduler")
    parser.add_argument("--mode", choices=['default', 'custom'], default='default',
                       help="Scheduling mode: default jobs or custom configuration")
    parser.add_argument("--config", help="Custom jobs configuration file (JSON)")
    parser.add_argument("--dry-run", action='store_true', help="Show scheduled jobs without running")
    
    args = parser.parse_args()
    
    # Create scheduler
    scheduler = MLDSScheduler()
    
    if args.mode == 'default':
        scheduler.setup_default_jobs()
    elif args.mode == 'custom' and args.config:
        with open(args.config, 'r') as f:
            custom_jobs = json.load(f)
        scheduler.setup_custom_jobs(custom_jobs)
    
    if args.dry_run:
        print("📅 Scheduled jobs:")
        jobs = schedule.get_jobs()
        for i, job in enumerate(sorted(jobs, key=lambda x: x.next_run)):
            next_run = job.next_run.strftime("%Y-%m-%d %H:%M:%S")
            print(f"  {i+1}. {job.job_func.__name__} at {next_run}")
        print(f"\nTotal: {len(jobs)} jobs scheduled")
    else:
        # Run scheduler
        scheduler.run()


if __name__ == "__main__":
    main()
