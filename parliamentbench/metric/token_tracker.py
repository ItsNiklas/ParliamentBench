"""Token usage tracker for LLM API calls."""

import json
import os
from datetime import datetime
from pathlib import Path


class TokenTracker:
    def __init__(self, output_dir: str = "runs/token_stats"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.stats_file = self.output_dir / f"tokens_{self.session_id}.jsonl"
        
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_requests = 0
        self.by_stage = {}
        self.model_name = os.environ.get("MODEL", "unknown")
    
    def track_response(self, response, stage: str = "unknown", player_name: str = "unknown"):
        if not hasattr(response, 'usage') or response.usage is None:
            return
        
        usage = response.usage
        self.total_prompt_tokens += usage.prompt_tokens
        self.total_completion_tokens += usage.completion_tokens
        self.total_requests += 1
        
        if stage not in self.by_stage:
            self.by_stage[stage] = {'prompt_tokens': 0, 'completion_tokens': 0, 'requests': 0}
        
        self.by_stage[stage]['prompt_tokens'] += usage.prompt_tokens
        self.by_stage[stage]['completion_tokens'] += usage.completion_tokens
        self.by_stage[stage]['requests'] += 1
        
        with open(self.stats_file, 'a') as f:
            f.write(json.dumps({
                'timestamp': datetime.now().isoformat(),
                'stage': stage,
                'player': player_name,
                'prompt_tokens': usage.prompt_tokens,
                'completion_tokens': usage.completion_tokens,
                'total_tokens': usage.total_tokens,
                'model': self.model_name
            }) + '\n')
    
    def save_summary(self):
        with open(self.output_dir / f"summary_{self.session_id}.json", 'w') as f:
            json.dump({
                'session_id': self.session_id,
                'model': self.model_name,
                'total_requests': self.total_requests,
                'total_prompt_tokens': self.total_prompt_tokens,
                'total_completion_tokens': self.total_completion_tokens,
                'total_tokens': self.total_prompt_tokens + self.total_completion_tokens,
                'by_stage': self.by_stage
            }, f, indent=2)
    
    def print_summary(self):
        total = self.total_prompt_tokens + self.total_completion_tokens
        print(f"\n{'='*60}")
        print(f"TOKEN USAGE: {total:,} tokens ({self.total_requests} requests)")
        print(f"  Prompt: {self.total_prompt_tokens:,} | Completion: {self.total_completion_tokens:,}")
        print(f"  Model: {self.model_name}")
        print(f"{'='*60}\n")


_tracker = None

def get_tracker():
    global _tracker
    if _tracker is None:
        _tracker = TokenTracker()
    return _tracker

def track_response(response, stage: str = "unknown", player_name: str = "unknown"):
    get_tracker().track_response(response, stage, player_name)
