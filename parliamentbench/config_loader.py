"""Configuration loader for Secret Hitler game."""
import os
import re
import yaml
from typing import Tuple


class Config:
    """Main configuration object."""
    def __init__(self):
        # Game settings
        self.players = 5
        self.output = "output.txt"
        self.log_level = "INFO"
        self.summary_path = "./runs"
        self.simple_mode = False
        self.cutoff_rounds = 0
        self.player_types = ["LLM", "CPU", "CPU", "CPU", "CPU"]
        self.enable_pre_vote_discussion = True
        self.enable_post_policy_discussion = True
        self.enable_reflection = True
        self.enable_rapid_assessment = True
        self.max_tokens = 1024
        self.request_timeout = 60
        self.max_api_retries = 3
        
        # LLM endpoints (nested dicts)
        self.llm = {
            "default": {"api_key": "", "base_url": "http://localhost:8080/v1/"},
            "llm_player": None,
            "basic_llm_player": None,
            "cpu_player": None,
            "rule_player": None,
            "random_player": None,
        }
        
        # Processing
        self.enable_parallel = True
        self.max_parallel_games = 4

        # Benchmark runner defaults
        self.benchmark_num_games = 100
        self.benchmark_role_schedule = "balanced"  # balanced|random|fixed
        self.benchmark_fixed_role = "liberal"
        self.benchmark_seed = None
        self.benchmark_run_name = None

        # Optional loaded-terms theme. Default None (= original
        # Secret Hitler terminology); set to "neutral" to remap all loaded
        # tokens to the Council / Reds / Blues / Speaker / Deputy theme.
        self.theme = None
        # When a theme is active, raise on a forbidden-token leak rather than
        # just logging it (also settable via the THEME_STRICT env var).
        self.theme_strict = False

        # Generation knobs for the LLM call path. Configurable so reasoning
        # on/off and retry tuning need no env vars or code edits.
        #
        # The three budget/retry knobs default to None, meaning "inherit the
        # game-level value", so a config without a `generation:` section keeps
        # its previous behaviour exactly. `effective_generation()` resolves
        # them.
        self.reasoning_enabled = True      # False => thinking-OFF bundle
        self.reasoning_effort = "low"      # reasoning effort when enabled: low|high
        self.max_retries = None            # None => game.max_api_retries
        self.completion_max_tokens = None  # None => game.max_tokens
        self.basic_max_tokens = None       # None => min(512, game.max_tokens)
    
    @staticmethod
    def _expand_env_vars(value):
        """Expand environment variables in string values with support for ${VAR:-default} syntax."""
        if not isinstance(value, str):
            return value
        
        # Handle ${VAR:-default} syntax
        def replace_with_default(match):
            var_name = match.group(1)
            default_value = match.group(2)
            return os.environ.get(var_name, default_value)
        
        # Replace ${VAR:-default} patterns
        value = re.sub(r'\$\{([^}:]+):-([^}]*)\}', replace_with_default, value)
        
        # Handle remaining $VAR or ${VAR} patterns
        value = os.path.expandvars(value)
        
        return value

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}
        
        config = cls()
        
        # Load game settings
        if 'game' in data:
            g = data['game']
            config.players = g.get('players', config.players)
            config.output = g.get('output', config.output)
            config.log_level = g.get('log_level', config.log_level)
            config.summary_path = g.get('summary_path', config.summary_path)
            config.simple_mode = g.get('simple_mode', config.simple_mode)
            config.cutoff_rounds = g.get('cutoff_rounds', config.cutoff_rounds)
            config.player_types = g.get('player_types', config.player_types)
            config.enable_pre_vote_discussion = g.get('enable_pre_vote_discussion', config.enable_pre_vote_discussion)
            config.enable_post_policy_discussion = g.get('enable_post_policy_discussion', config.enable_post_policy_discussion)
            config.enable_reflection = g.get('enable_reflection', config.enable_reflection)
            config.enable_rapid_assessment = g.get('enable_rapid_assessment', config.enable_rapid_assessment)
            config.max_tokens = g.get('max_tokens', config.max_tokens)
            config.request_timeout = g.get('request_timeout', config.request_timeout)
            config.max_api_retries = g.get('max_api_retries', config.max_api_retries)
            config.theme = g.get('theme', config.theme)
            config.theme_strict = g.get('theme_strict', config.theme_strict)

        # Load LLM settings
        if 'llm' in data:
            llm = data['llm']
            
            if 'default' in llm:
                config.llm['default'] = {
                    'api_key': config._expand_env_vars(llm['default'].get('api_key', '')),
                    'base_url': config._expand_env_vars(llm['default'].get('base_url', 'http://localhost:8080/v1/')),
                    'model': llm['default'].get('model'),
                }
            
            if 'llm_player' in llm:
                advanced_cfg = llm['llm_player'].get('advanced')
                if advanced_cfg:
                    advanced_cfg = {
                        'api_key': config._expand_env_vars(advanced_cfg.get('api_key', '')),
                        'base_url': config._expand_env_vars(advanced_cfg.get('base_url', 'http://localhost:8080/v1/')),
                        'model': advanced_cfg.get('model'),
                    }
                standard_cfg = llm['llm_player'].get('standard')
                if standard_cfg:
                    standard_cfg = {
                        'api_key': config._expand_env_vars(standard_cfg.get('api_key', '')),
                        'base_url': config._expand_env_vars(standard_cfg.get('base_url', 'http://localhost:8080/v1/')),
                        'model': standard_cfg.get('model'),
                    }
                config.llm['llm_player'] = {
                    'api_key': config._expand_env_vars(llm['llm_player'].get('api_key')),
                    'base_url': config._expand_env_vars(llm['llm_player'].get('base_url')),
                    'model': llm['llm_player'].get('model'),
                    'advanced_player_index': llm['llm_player'].get('advanced_player_index', 0),
                    'advanced': advanced_cfg,
                    'standard': standard_cfg,
                }
            
            if 'basic_llm_player' in llm:
                config.llm['basic_llm_player'] = {
                    'api_key': config._expand_env_vars(llm['basic_llm_player'].get('api_key', '')),
                    'base_url': config._expand_env_vars(llm['basic_llm_player'].get('base_url', 'http://localhost:8080/v1/')),
                    'model': llm['basic_llm_player'].get('model'),
                }
            
            if 'cpu_player' in llm:
                config.llm['cpu_player'] = {
                    'api_key': config._expand_env_vars(llm['cpu_player'].get('api_key', '')),
                    'base_url': config._expand_env_vars(llm['cpu_player'].get('base_url', 'http://localhost:8080/v1/')),
                    'model': llm['cpu_player'].get('model'),
                }
            
            if 'rule_player' in llm:
                config.llm['rule_player'] = {
                    'api_key': config._expand_env_vars(llm['rule_player'].get('api_key', '')),
                    'base_url': config._expand_env_vars(llm['rule_player'].get('base_url', 'http://localhost:8080/v1/')),
                    'model': llm['rule_player'].get('model'),
                }
            
            if 'random_player' in llm:
                config.llm['random_player'] = {
                    'api_key': config._expand_env_vars(llm['random_player'].get('api_key', '')),
                    'base_url': config._expand_env_vars(llm['random_player'].get('base_url', 'http://localhost:8080/v1/')),
                    'model': llm['random_player'].get('model'),
                }
        
        # Load processing settings
        if 'processing' in data:
            p = data['processing']
            config.enable_parallel = p.get('enable_parallel', config.enable_parallel)
            config.max_parallel_games = p.get('max_parallel_games', config.max_parallel_games)

        # Load benchmark settings
        if 'benchmark' in data:
            b = data['benchmark']
            config.benchmark_num_games = b.get('num_games', config.benchmark_num_games)
            config.benchmark_role_schedule = b.get('role_schedule', config.benchmark_role_schedule)
            config.benchmark_fixed_role = b.get('fixed_role', config.benchmark_fixed_role)
            config.benchmark_seed = b.get('seed', config.benchmark_seed)
            config.benchmark_run_name = b.get('run_name', config.benchmark_run_name)

        # Load generation settings (LLM call knobs)
        if 'generation' in data:
            gen = data['generation']
            config.reasoning_enabled = gen.get('reasoning_enabled', config.reasoning_enabled)
            config.reasoning_effort = gen.get('reasoning_effort', config.reasoning_effort)
            config.max_retries = gen.get('max_retries', config.max_retries)
            config.completion_max_tokens = gen.get('max_tokens', config.completion_max_tokens)
            config.basic_max_tokens = gen.get('basic_max_tokens', config.basic_max_tokens)

        return config

    def effective_generation(self) -> dict:
        """Resolve the generation knobs against the game-level fallbacks.

        Returns concrete values for every knob, so callers never have to deal
        with the "None means inherit" sentinel.
        """
        return {
            'reasoning_enabled': self.reasoning_enabled,
            'reasoning_effort': self.reasoning_effort,
            'max_retries': (
                self.max_retries if self.max_retries is not None else self.max_api_retries
            ),
            'completion_max_tokens': (
                self.completion_max_tokens
                if self.completion_max_tokens is not None
                else self.max_tokens
            ),
            'basic_max_tokens': (
                self.basic_max_tokens
                if self.basic_max_tokens is not None
                else min(512, self.max_tokens)
            ),
        }
    
    def get_llm_endpoint(self, player_type: str, player_index: int = 0) -> Tuple[str, str, str]:
        """Get API key, base URL, and model override for a specific player type and index.
        
        Returns:
            Tuple of (api_key, base_url, model) where model may be None if not overridden.
        """
        player_type = player_type.upper()
        
        if player_type == "LLM" and self.llm['llm_player']:
            lp = self.llm['llm_player']
            # Check if this is the advanced player
            if player_index == lp.get('advanced_player_index', 0) and lp.get('advanced'):
                return (lp['advanced']['api_key'], lp['advanced']['base_url'], lp['advanced'].get('model'))
            # Standard LLM player
            elif lp.get('standard'):
                return (lp['standard']['api_key'], lp['standard']['base_url'], lp['standard'].get('model'))
            # Fallback to llm_player base config
            elif lp.get('api_key'):
                return (lp['api_key'], lp.get('base_url') or self.llm['default']['base_url'], lp.get('model'))
        
        elif player_type == "BASICLLM" and self.llm['basic_llm_player']:
            return (self.llm['basic_llm_player']['api_key'], self.llm['basic_llm_player']['base_url'], self.llm['basic_llm_player'].get('model'))
        
        elif player_type == "CPU" and self.llm['cpu_player']:
            return (self.llm['cpu_player']['api_key'], self.llm['cpu_player']['base_url'], self.llm['cpu_player'].get('model'))
        
        elif player_type == "RULE":
            if self.llm['rule_player']:
                return (self.llm['rule_player']['api_key'], self.llm['rule_player']['base_url'], self.llm['rule_player'].get('model'))
            # Fallback to CPU config
            elif self.llm['cpu_player']:
                return (self.llm['cpu_player']['api_key'], self.llm['cpu_player']['base_url'], self.llm['cpu_player'].get('model'))
        
        elif player_type == "RANDOM":
            if self.llm['random_player']:
                return (self.llm['random_player']['api_key'], self.llm['random_player']['base_url'], self.llm['random_player'].get('model'))
            # Fallback to CPU config
            elif self.llm['cpu_player']:
                return (self.llm['cpu_player']['api_key'], self.llm['cpu_player']['base_url'], self.llm['cpu_player'].get('model'))
        
        # Default fallback
        return (self.llm['default']['api_key'], self.llm['default']['base_url'], self.llm['default'].get('model'))
