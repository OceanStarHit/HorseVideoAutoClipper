import time
from engine.processor import Processor


class Pipeline:
    def __init__(
        self,
        video_paths,
        output_dir,
        log_panel,
        progress_bar,
        on_complete=None,
    ):
        self.video_paths = video_paths
        self.output_dir = output_dir
        self.log = log_panel.log
        self.progress = progress_bar
        self.on_complete = on_complete

        self.processor = Processor(
            output_dir=self.output_dir,
            logger=self.log,
            imgsz=320   # try 512 or even 416 if still slow
        )

    # -------------------------------------------------------
    def run(self):
        total = len(self.video_paths)
        self.progress["maximum"] = total
        self.progress["value"] = 0

        self.log(f"🚀 Starting batch processing ({total} video(s))")
        self.log(f"📁 Output directory: {self.output_dir}")
        self.log("-" * 60)

        start_batch_time = time.time()
        success_count = 0
        reject_count = 0
        error_count = 0

        for idx, video_path in enumerate(self.video_paths, start=1):
            video_start = time.time()
            self.log(f"🎬 [{idx}/{total}] Processing video: {video_path}")

            try:
                # ---------------------------------------------------
                # Processing video
                # ---------------------------------------------------
                success = self.processor.run(video_path)
                if success:
                    success_count += 1
                else: 
                    reject_count += 1

            except Exception as e:
                self.log(f"     💥 Error while processing video: {video_path}")
                self.log(f"        {e}")
                error_count += 1

            elapsed = time.time() - video_start
            self.log(f"     ⏱ Video processing time: {elapsed:.2f}s")
            self.log("-" * 60)

            self.progress["value"] = idx

        # ---------------------------------------------------
        # Batch summary
        # ---------------------------------------------------
        total_time = time.time() - start_batch_time

        self.log("🏁 Batch processing completed")
        self.log(
            f"✔ Successful clips: {success_count} | "
            f"⚠ Rejected: {reject_count} | "
            f"❌Errors: {error_count}"
        )
        self.log(f"⏱ Total processing time: {total_time:.2f}s")

        if self.on_complete:
            try:
                self.on_complete()
            except Exception:
                pass
