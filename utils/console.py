from colorama import init as colorama_init, Fore, Style
colorama_init(autoreset=True)

def _c(txt: str, color: str = Fore.WHITE, bright: bool = False) -> str:
    """Small helper for colored console logs."""
    return f"{Style.BRIGHT if bright else ''}{color}{txt}{Style.RESET_ALL}"
