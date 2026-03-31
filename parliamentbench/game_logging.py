import logging
import os
import re
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown
from io import StringIO

# Regular expression to remove rich markup tags
RICH_MARKUP_PATTERN = re.compile(r'\[([a-z0-9_\.]+)\](.*?)\[\/\1\]')

# Configure custom rich theme
custom_theme = Theme({
    "liberal": "bold bright_cyan",
    "fascist": "bold bright_red",
    "hitler": "bold magenta",
    "info": "bright_green",
    "warning": "yellow",
    "error": "bold red",
    "critical": "bold white on red",
    "debug": "dim cyan",
    "policy.liberal": "blue on bright_white",
    "policy.fascist": "white on red",
    "action": "bold magenta",
    "vote.ja": "green",
    "vote.nein": "red",
    "player.dead": "dim strike",
    "player.alive": "bold",
    "game.turn": "bold yellow on blue",
})

# Create rich console
console = Console(theme=custom_theme)

# Configure logging with rich
logger = logging.getLogger("ParliamentGame")
logger.setLevel(logging.DEBUG)  # Default level

# Create rich handler for console output
# Set level to WARNING so INFO messages don't appear in terminal (avoiding duplication)
rich_handler = RichHandler(
    console=console,
    show_path=False,
    omit_repeated_times=True,
    markup=True,
    rich_tracebacks=True,
    tracebacks_show_locals=False,
)
rich_handler.setLevel(logging.WARNING)  # Only show WARNING and above in terminal

# File handler will be initialized later when the log path is known
file_handler = None

def set_log_level(level):
    """Set the log level for both logger and console handler.
    
    Args:
        level: logging level (e.g., logging.INFO, logging.DEBUG)
    """
    logger.setLevel(level)
    # Update the rich handler to show messages at the configured level
    for handler in logger.handlers:
        if isinstance(handler, RichHandler):
            handler.setLevel(level)

# Create a clean formatter for file logs without rich markup
class CleanFormatter(logging.Formatter):
    """Custom formatter that removes rich markup from log messages"""
    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)
    
    def format(self, record):
        # First format the record using the parent formatter
        formatted = super().format(record)
        
        # Remove rich markup tags
        clean_text = RICH_MARKUP_PATTERN.sub(r'\2', formatted)
        
        # Remove other styling codes that might be in the text
        clean_text = re.sub(r'\[bold\]|\[\/bold\]', '', clean_text)
        clean_text = re.sub(r'\[italic\]|\[\/italic\]', '', clean_text)
        clean_text = re.sub(r'\[.*?\]', '', clean_text)
        
        return clean_text

# Create clean formatter for files
file_formatter = CleanFormatter("%(asctime)s - %(levelname)s - %(message)s")

# Set up logger with just the rich handler initially
logger.addHandler(rich_handler)

# Remove default handlers
for handler in logger.handlers[:]:
    if isinstance(handler, logging.StreamHandler) and not isinstance(handler, RichHandler):
        logger.removeHandler(handler)

def init_file_logger(output_path=None):
    """Initialize the file logger with the provided output path"""
    global file_handler
    
    # If a file handler already exists, remove it
    if file_handler is not None and file_handler in logger.handlers:
        logger.removeHandler(file_handler)
        
    # Use the output path for logging if provided, otherwise use default
    log_file = output_path if output_path else "hitler_game.log"
    
    # Create directory for log file if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # Create and add the file handler with the clean formatter
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.INFO)  # Ensure all INFO messages go to the file
    logger.addHandler(file_handler)
    
    # Use console.print for visible feedback instead of logger.info
    console.print(f"[info]Logging to file: {log_file}[/info]")
    # Still log to file but won't show in terminal since it's INFO level
    logger.info(f"Logging to file: {log_file}")

def strip_rich_markup(text):
    """Remove rich markup from text for plain logging"""
    # Remove rich markup tags
    clean_text = RICH_MARKUP_PATTERN.sub(r'\2', text)
    
    # Remove other styling codes
    clean_text = re.sub(r'\[bold\]|\[\/bold\]', '', clean_text)
    clean_text = re.sub(r'\[italic\]|\[\/italic\]', '', clean_text)
    clean_text = re.sub(r'\[.*?\]', '', clean_text)
    
    return clean_text

def console_print_and_log(text, level="info"):
    """Print to console with rich formatting and also log to file without formatting"""
    # Print to console with rich formatting
    console.print(text)
    
    # Log to file without rich formatting
    plain_text = strip_rich_markup(str(text))
    
    # Use the appropriate logging level
    if level.lower() == "debug":
        logger.debug(plain_text)
    elif level.lower() == "warning":
        logger.warning(plain_text)
    elif level.lower() == "error":
        logger.error(plain_text)
    elif level.lower() == "critical":
        logger.critical(plain_text)
    else:
        logger.info(plain_text)

# Helper functions for pretty output
def format_player_name(player):
    """Format a player name with styling based on their status"""
    if player is None:
        return "None"
        
    if hasattr(player, "is_dead"):
        if player.is_dead:
            return f"[player.dead]{player.name}[/player.dead]"
        return f"[player.alive]{player.name}[/player.alive]"
    return str(player)


def format_player_with_role(player):
    """Format a player name with role-specific color"""
    if player is None:
        return "None"
        
    player_name = player.name
    if player.is_hitler:
        return f"[hitler]{player_name}[/hitler]"
    elif player.is_fascist:
        return f"[fascist]{player_name}[/fascist]"
    else:
        return f"[liberal]{player_name}[/liberal]"


def format_policy(policy):
    """Format a policy with appropriate styling"""
    if policy is None:
        return "None"
        
    if hasattr(policy, 'type'):
        if policy.type == "liberal":
            return "[policy.liberal]Liberal[/policy.liberal]"
        elif policy.type == "fascist":
            return "[policy.fascist]Fascist[/policy.fascist]"
    return str(policy)


def format_role(role):
    """Format a role with appropriate styling"""
    if role is None:
        return "None"
        
    if hasattr(role, 'role'):
        if role.role == "liberal":
            return "[liberal]Liberal[/liberal]"
        elif role.role == "fascist":
            return "[fascist]Fascist[/fascist]"
        elif role.role == "hitler":
            return "[hitler]Hitler[/hitler]"
    return str(role)


def format_vote(vote):
    """Format a vote with appropriate styling"""
    if vote:
        return "[vote.ja]JA[/vote.ja]"
    return "[vote.nein]NEIN[/vote.nein]"


def display_game_header():
    """Display a header for the game"""
    header = Panel.fit("      [bold yellow]SECRET HITLER[/bold yellow]      ", 
                      subtitle="[italic]May the best faction win. [/italic]")
    console.print(header)
    logger.info("=== SECRET HITLER GAME STARTING ===")


def display_game_start(player_count):
    """Display a message at game start"""
    panel = Panel.fit("   [bold]Secret Hitler LLM Simulator[/bold]   ", 
                     subtitle=f"Starting game with {player_count} players")
    console.print(panel)
    logger.info(f"Starting game with {player_count} players")


def display_turn_header(turn_number):
    """Display a header for a new turn"""
    console.print(Rule(f"[game.turn]Turn {turn_number}[/game.turn]"))
    logger.info(f"--- TURN {turn_number} ---")
    # return console.input("[bright_white]Press Enter to start the next turn...[/bright_white]")
    return


def display_game_status(game_state):
    """Display a pretty game status panel"""
    table = Table(show_header=False)
    table.add_column("Key")
    table.add_column("Value")
    
    table.add_row("Liberal Policies", f"{game_state.liberal_track}/{5}")  # LIBERAL_POLICIES_TO_WIN
    table.add_row("Fascist Policies", f"{game_state.fascist_track}/{6}")  # FASCIST_POLICIES_TO_WIN
    table.add_row("Failed Votes", str(game_state.failed_votes))
    
    president_text = format_player_name(game_state.president) if game_state.president else "None"
    chancellor_text = format_player_name(game_state.chancellor) if game_state.chancellor else "None"
    
    table.add_row("President", president_text)
    table.add_row("Chancellor", chancellor_text)
    
    console.print(Panel(table, title="Game Status", border_style="bright_blue"))
    
    # Log a plain text version of the game status
    pres_name = game_state.president.name if game_state.president else "None"
    chanc_name = game_state.chancellor.name if game_state.chancellor else "None"
    logger.info(f"GAME STATUS: Liberal Policies: {game_state.liberal_track}/5, Fascist Policies: {game_state.fascist_track}/6, Failed Votes: {game_state.failed_votes}, President: {pres_name}, Chancellor: {chanc_name}")


def display_chancellor_nomination(president, chancellor):
    """Display chancellor nomination message"""
    # Get president and chancellor names with role-specific coloring
    president_name = format_player_with_role(president)
    chancellor_name = format_player_with_role(chancellor)
    
    message = f"President {president_name} nominates {chancellor_name} as Chancellor"
    console.print(message)
    
    # Include the roles in the log file
    president_role = "(Hitler)" if president.is_hitler else "(Fascist)" if president.is_fascist else "(Liberal)"
    chancellor_role = "(Hitler)" if chancellor.is_hitler else "(Fascist)" if chancellor.is_fascist else "(Liberal)"
    
    logger.info(f"President {president.name} {president_role} nominates {chancellor.name} {chancellor_role} as Chancellor")


def display_progress_with_spinner(description, total=1):
    """Display a progress bar with spinner"""
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    )
    logger.info(description)
    return progress, progress.add_task(description + '\n', total=total)


def display_vote_results(vote_count, passed):
    """Display vote results with appropriate styling"""
    if passed:
        msg = f"Vote passed: {vote_count[0]} [vote.ja]JA[/vote.ja], {vote_count[1]} [vote.nein]NEIN[/vote.nein]"
        console.print(f"[green]{msg}[/green]")
        logger.info(f"Vote passed: {vote_count[0]} JA, {vote_count[1]} NEIN")
    else:
        msg = f"Vote failed: {vote_count[0]} [vote.ja]JA[/vote.ja], {vote_count[1]} [vote.nein]NEIN[/vote.nein]"
        console.print(f"[yellow]{msg}[/yellow]")
        logger.info(f"Vote failed: {vote_count[0]} JA, {vote_count[1]} NEIN")


def display_failed_votes(failed_votes):
    """Display the current failed vote count"""
    console.print(f"[yellow]Election tracker now at {failed_votes}/3[/yellow]")
    logger.info(f"Election tracker now at {failed_votes}/3")


def display_election_tracker_full():
    """Display message when election tracker is full"""
    console.print("[bold red]❗ Election tracker reached 3! Enacting top policy automatically[/bold red]")
    logger.warning("Election tracker reached 3! Enacting top policy automatically")


def display_policy_enacted(policy):
    """Display a message when a policy is enacted"""
    policy_text = format_policy(policy)
    console.print(f"[bold]Policy enacted: {policy_text}[/bold]")
    logger.info(f"Policy enacted: {policy.type.capitalize()}")


def display_veto():
    """Display a message when veto power is used"""
    console.print("[bold yellow]⚠️ Veto power used![/bold yellow]")
    logger.info("Veto power used!")


def display_special_action(action):
    """Display a message when a special action is triggered"""
    console.print(f"[action]Special action triggered: {action.upper()}[/action]")
    logger.info(f"Special action triggered: {action.upper()}")


def display_player_executed(killed_player, president):
    """Display a message when a player is executed"""
    console.print(f"[bold red]💀 {format_player_name(killed_player)} has been executed by President {format_player_name(president)} 💀[/bold red]")
    logger.info(f"{killed_player.name} has been executed by President {president.name}")


def display_player_investigated(president, inspected):
    """Display a message when a player is investigated"""
    console.print(f"President {format_player_name(president)} investigated {format_player_name(inspected)}'s loyalty")
    logger.info(f"President {president.name} investigated {inspected.name}'s loyalty")


def display_next_president_chosen(president, chosen):
    """Display a message when next president is chosen"""
    console.print(f"President {format_player_name(president)} chose {format_player_name(chosen)} to be the next president")
    logger.info(f"President {president.name} chose {chosen.name} to be the next president")


def display_policy_view(president):
    """Display a message when president views policies"""
    console.print(f"President {format_player_name(president)} viewed the top 3 policies")
    logger.info(f"President {president.name} viewed the top 3 policies")


def display_policy_table(policies, title, include_indices=False):
    """Display a table of policies"""
    policy_table = Table(title=title)
    
    if include_indices:
        policy_table.add_column("#", style="dim")
        policy_table.add_column("Policy", style="bright_white")
        
        for i, policy in enumerate(policies):
            policy_table.add_row(str(i), format_policy(policy))
    else:
        policy_table.add_column("Policy", style="bright_white")
        
        for policy in policies:
            policy_table.add_row(format_policy(policy))
    
    console.print(policy_table)
    
    # Log plaintext version of the policy table
    policy_types = [p.type.capitalize() for p in policies]
    logger.info(f"{title}: {', '.join(policy_types)}")


def display_vote_table(players):
    """Display a table of player votes"""
    # Create a vote table
    vote_table = Table(title="Government Vote")
    vote_table.add_column("Player", style="cyan")
    vote_table.add_column("Vote", style="yellow")
    
    # Plain text log of votes for the log file
    vote_log = "Votes: "
    
    for player, vote in players:
        if not player.is_dead:
            vote_str = "[vote.ja]JA[/vote.ja]" if vote else "[vote.nein]NEIN[/vote.nein]"
            vote_table.add_row(format_player_name(player), vote_str)
            vote_log += f"{player.name}: {'JA' if vote else 'NEIN'}, "
    
    console.print(vote_table)
    logger.info(vote_log.rstrip(", "))


def display_player_vote(player, vote):
    """Display a player's vote"""
    if vote:
        console.print(Panel(
            f"[vote.ja]{player.name} votes JA[/vote.ja]", 
            title=player.name, 
            subtitle="Government Vote",
            border_style="green"
        ))
        logger.info(f"{player.name} votes JA")
    else:
        console.print(Panel(
            f"[vote.nein]{player.name} votes NEIN[/vote.nein]", 
            title=player.name, 
            subtitle="Government Vote",
            border_style="red"
        ))
        logger.info(f"{player.name} votes NEIN")


def clean_markdown_content(content):
    """Clean markdown content for plain text logging"""
    # Remove markdown headers
    cleaned = re.sub(r'#+\s+', '', content)
    # Remove markdown emphasis
    cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', cleaned)
    cleaned = re.sub(r'\*(.*?)\*', r'\1', cleaned)
    # Remove markdown lists
    cleaned = re.sub(r'^\s*[-*+]\s+', '- ', cleaned, flags=re.MULTILINE)
    # Remove markdown links
    cleaned = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', cleaned)
    # Remove code blocks and inline code
    cleaned = re.sub(r'```.*?```', '[CODE BLOCK]', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'`(.*?)`', r'\1', cleaned)
    # Remove other markdown formatting
    cleaned = re.sub(r'~{2}(.*?)~{2}', r'\1', cleaned)  # strikethrough
    
    return cleaned
    

def display_player_reasoning(player, content, title, border_style="blue"):
    """Display a player's reasoning in a panel"""
    # Get player name with role-specific coloring
    player_name = format_player_with_role(player)
    
    # Update title to include role-colored player name if the title contains the player's name
    if player.name in title:
        title = title.replace(player.name, player_name)
    
    console.print(Panel(
        Markdown(content),
        title=title,
        border_style=border_style
    ))
    
    # Clean the content for logging and log the full content
    clean_content = clean_markdown_content(content)
    
    # Include the player's role in the log file
    role_text = ""
    if player.is_hitler:
        role_text = "(Hitler)"
    elif player.is_fascist:
        role_text = "(Fascist)"
    else:
        role_text = "(Liberal)"
        
    logger.info(f"{title} {role_text}:\n{clean_content}")

def display_player_inner_monologue(player, content, title, border_style="blue"):
    """Display a player's inner monologue in a panel"""
    # Get player name with role-specific coloring
    player_name = format_player_with_role(player)

    # Update title to include role-colored player name if the title contains the player's name
    if player.name in title:
        title = title.replace(player.name, player_name)

    console.print(Panel(
        Markdown(content),
        title=title,
        border_style=border_style
    ))

    # Clean the content for logging and log the full content
    clean_content = clean_markdown_content(content)

    # Include the player's role in the log file
    role_text = ""
    if player.is_hitler:
        role_text = "(Hitler)"
    elif player.is_fascist:
        role_text = "(Fascist)"
    else:
        role_text = "(Liberal)"

    logger.info(f"{title} {role_text}:\n{clean_content}")


def display_player_discussion(player, content):
    """Display a player's discussion comments"""
    # Add role-specific color to player name
    player_name = player.name
    if player.is_hitler:
        player_name = f"[hitler]{player_name}[/hitler]"
    elif player.is_fascist:
        player_name = f"[fascist]{player_name}[/fascist]"
    else:
        # Liberal
        player_name = f"[liberal]{player_name}[/liberal]"
        
    console.print(
        Panel(
            Markdown(content),
            title=f"{player_name}'s comments",
            border_style="bright_green"
        )
    )
    
    # Clean the content for logging and log the full content
    clean_content = clean_markdown_content(content)
    
    # Include the player's role in the log file
    role_text = ""
    if player.is_hitler:
        role_text = "(Hitler)"
    elif player.is_fascist:
        role_text = "(Fascist)"
    else:
        role_text = "(Liberal)"
        
    logger.info(f"{player.name}'s {role_text} comments:\n{clean_content}")


def display_info_message(message):
    """Display an info message"""
    console.print(f"[info]{message}[/info]")
    # Also log to file
    logger.info(strip_rich_markup(message))


def display_game_over(result_type):
    """Display game over message based on result type"""
    if result_type == "hitler_chancellor":
        result_text = "Fascists win by electing Hitler as Chancellor!"
        console.print(Panel(
            f"[bold red]{result_text}[/bold red]",
            title="Game Over",
            border_style="red"
        ))
        logger.info(f"GAME OVER: {result_text}")
    elif result_type == "liberal_policy":
        result_text = "Liberals win by enacting 5 Liberal policies!"
        console.print(Panel(
            f"[bold blue]{result_text}[/bold blue]",
            title="Game Over",
            border_style="blue"
        ))
        logger.info(f"GAME OVER: {result_text}")
    elif result_type == "fascist_policy":
        result_text = "Fascists win by enacting 6 Fascist policies!"
        console.print(Panel(
            f"[bold red]{result_text}[/bold red]",
            title="Game Over",
            border_style="red"
        ))
        logger.info(f"GAME OVER: {result_text}")
    elif result_type == "hitler_killed":
        result_text = "Liberals win by assassinating Hitler!"
        console.print(Panel(
            f"[bold blue]{result_text}[/bold blue]",
            title="Game Over",
            border_style="blue"
        ))
        logger.info(f"GAME OVER: {result_text}")


def with_status(message):
    """Context manager for showing status messages"""
    logger.info(strip_rich_markup(message))
    return console.status(message + '\n', spinner="dots")


def format_state_for_display(game_state):
    """Format game state for display"""
    buffer = StringIO()
    temp_console = Console(file=buffer, width=80)

    # Create a table for the game state
    table = Table(show_header=False, title="Game State")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="yellow")
    
    # Add rows for the game state
    table.add_row("Liberal policies", f"[liberal]{game_state.liberal_track}[/liberal]")
    table.add_row("Fascist policies", f"[fascist]{game_state.fascist_track}[/fascist]")
    table.add_row("Failed votes", str(game_state.failed_votes))
    
    # Add current government
    pres = format_player_name(game_state.president) if game_state.president else "None"
    chanc = format_player_name(game_state.chancellor) if game_state.chancellor else "None"
    ex_pres = format_player_name(game_state.ex_president) if game_state.ex_president else "None"
    
    table.add_row("President", pres)
    table.add_row("Chancellor", chanc)
    table.add_row("Former President", ex_pres)
    
    # Add recent policy
    recent_policy = format_policy(game_state.most_recent_policy) if game_state.most_recent_policy else "None"
    table.add_row("Most recent policy", recent_policy)
    
    # Add votes if available
    if game_state.last_votes:
        votes_formatted = ", ".join([format_vote(vote) for vote in game_state.last_votes])
        table.add_row("Last votes", votes_formatted)
    
    # Add veto status
    table.add_row("Veto power", "[green]Available[/green]" if game_state.fascist_track >= 5 else "[red]Unavailable[/red]")
    
    # Print players
    player_table = Table(title="Players")
    player_table.add_column("Name", style="cyan")
    player_table.add_column("Status", style="green")
    
    for player in game_state.players:
        status = "[red]Dead[/red]" if player.is_dead else "[green]Alive[/green]"
        player_table.add_row(player.name, status)

    # Write tables to buffer
    temp_console.print(table)
    temp_console.print(player_table)
    
    # Return the string representation
    return buffer.getvalue()