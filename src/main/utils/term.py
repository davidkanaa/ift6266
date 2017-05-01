import sys
from inspect import getframeinfo

from termcolor import cprint


def error(text, currentframe):
    """
    Prints the given `text` with the formatting for errors.

    Parameters
    ----------
    text : str
        Text to print
    currentframe : Frame object
        Place where the error took place
    """

    info = getframeinfo(currentframe)

    cprint("[!] File : {}".format(info.filename).ljust(80), "red", attrs=["bold"])
    cprint("[!] Line : {}".format(info.lineno).ljust(80), "red", attrs=["bold"])
    cprint("[!] {}".format(text).ljust(80), "red", attrs=["bold"])
    sys.exit()


def warning(text):
    """
    Prints the given `text` with the formatting for warnings.

    Parameters
    ----------
    text : str
        Text to print
    """

    cprint("[!] {}".format(text).ljust(80), "yellow", attrs=["bold"])


def debug(text):
    """
    Prints the text as bold yellow string for debugging

    Parameters
    ----------
    text : str
         Text to print

    Returns
    -------
    Nothing
    """

    print()
    cprint(text, "yellow", attrs=["bold"])


def progress(text):
    """
    Prints the text as yellow string with return for loading display.

    Parameters
    ----------
    text : str
         Text to print

    Returns
    -------
    Nothing
    """

    cprint("[*] {}".format(text).ljust(80), "yellow", end="\r")


def success(text):
    """
    Prints the text as cyan string with return for completion display.

    Parameters
    ----------
    text : str
         Text to print

    Returns
    -------
    Nothing
    """

    cprint("[+] {}".format(text).ljust(80), "cyan")
