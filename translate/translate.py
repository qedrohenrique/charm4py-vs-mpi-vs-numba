import secrets
import time

from deep_translator import GoogleTranslator
from charm4py import coro, charm, Chare, Group
import click

@coro
def translate_block(enumerated_text):
    idx, text = enumerated_text
    if len(text) == 0:  # Can happen because of EOF
        return text

    # sanitized = sanitize_text(text)
    translated = GoogleTranslator(source=IN_LANG, target=OUT_LANG).translate(text)
    print(f"Current line: {idx} - {round(time.time() - T0, 3)}s")

    return translated


@coro
def sanitize_text(text):
    sanitized_text = text.replace("\n", " ")
    if sanitized_text.startswith(" "):
        sanitized_text = sanitized_text[1:]
    if sanitized_text.endswith(" "):
        sanitized_text = sanitized_text[:-1]
    return sanitized_text


def join_with_blank_lines(text):
    return '. \n'.join(text)


def join_with_space(text):
    return '. '.join(text)


@click.command()
@click.option('--in_path', help='Input file you want to translate.')
@click.option('--out_path', default='out', help='Output file to save translation.')
@click.option('--in_lang', default='auto', help='Language to be translated.')
@click.option('--out_lang', default='pt', help='Language to translate in.')
def translate(in_path, out_path, in_lang, out_lang):
    """Program that translate a text with at least to phrases using Charm4Py."""

    try:
        f = open(in_path, 'r')
        input_text = f.read()
        iterable_text = input_text.split('.')
        f.close()
    except TypeError:
        print(f"Cannot find file to translate.")
        exit()

    charm.thisProxy.updateGlobals({'IN_LANG': in_lang}, awaitable=True).get()
    charm.thisProxy.updateGlobals({'OUT_LANG': out_lang}, awaitable=True).get()

    print(f"Translating {len(iterable_text)} lines.\n")

    t0 = time.time()
    charm.thisProxy.updateGlobals({'T0': t0}, awaitable=True).get()

    # Parallel work
    result = charm.pool.map(translate_block, list(enumerate(iterable_text)))

    print('')
    print('Elapsed time: ', round(time.time() - t0, 3), 's', sep='')

    short_hash = secrets.token_hex(nbytes=16)[:4]
    out_filename = f"{out_path}_{short_hash}"

    try:
        out_file = open(out_filename, 'w')
        out_file.write(join_with_space(result))
    except TypeError:
        print(f"Cannot write in {out_filename}.")
        print(join_with_space(result))
        exit()


def execute(args):
    print('')
    translate()


global IN_LANG
global OUT_LANG
global T0
charm.start(execute)
