"""Predefined character sets for ASCII art conversion."""

CHAR_SETS = {
    'standard': {
        'chars': " .:-=+*#%@",
        'name': 'Standard ASCII'
    },
    'standard2': {
        'chars': " ?=#&0%@",
        'name': 'Standard2 ASCII'
    },
    'standard3': {
        'chars': " ';-~|}/+=",
        'name': 'Standard3 ASCII'
    },
    'standard4': {
        'chars': " _*!~)(+^#&$%@",
        'name': 'Standard4 ASCII'
    },
    'standard5': {
        'chars': " `-~+#@",
        'name': 'Standard5 ASCII'
    },
    'standard6': {
        'chars': " ¨'³•µðEÆ",
        'name': 'Standard6 ASCII'
    },
    'standard7': {
        'chars': " `.,-:~;+*#%$@",
        'name': 'Standard7 ASCII'
    },    
    'standard_alt': {
        'chars': " .,:ilwW",
        'name': 'Standard ASCII Alternative'
    },
    'complex': {
        'chars': " `.',:^\";*!²¤/r(?+¿cLª7t1fJCÝy¢zF3±%S2kñ5AZXG$À0Ãm&Q8#RÔßÊNBåMÆØ@¶",
        'name': 'Complex ASCII '
    },
    'complex_alt': {
        'chars': " `.-':_,^=;><+!rc*/z?sLTv)J7(|Fi\{C\}fI31tlu[neoZ5Yxjya]2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@",
        'name': 'Complex ASCII Alternative'
    },
    'fine': {
        'chars': " `^\",:;Il!i~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$",
        'name': 'Fine Detail ASCII'
    },
    'runic': {
        'chars': " ᛫ᛌᚲ᛬ᛍᛵᛁᛊᚿᚽᚳᚪᚮᚩᚰᚨᛏᚠᚬᚧᛪᚫᚷᛱᚱᚢᛒᚤᛄᚣᚻᛖᛰᚸᚥᛞᛤᛥ",
        'name': 'Runic'
    },
    'box': {
        'chars': " ╶╴┈┄╌─┊╺┆└╸╭╰┉┬│╾┅┍┕┭━╘┎╱╲┖┰┯┋┸┇┞┗╙┱╧╀┹┢┡┳╅┻╃┃╚┠╈╇╳╂╦┣╩╉║╋╫╠╬",
        'name': 'Box Drawings'
    },
    'blocks': {
        'chars': " ▏▎▍▌▋▊▉█",
        'name': 'Block Elements '
    },
    'blocks_alt': {
        'chars': "  ▏▁░▂▖▃▍▐▒▀▞▚▌▅▆▊▓▇▉█",
        'name': 'Block Elements Alternative'
    },
    'geo': {
        'chars': " ◜◞◟◦◃◠▿▹▱◌▵◅▭▸◁△◹▽▫▷▯□◯◄▰◫◊◮◎◈◖◭◗▬◤▪▼◑◍▮◒◐▤◉▧▨◕◛◚▣▦●▩■◘◙",
        'name': 'Geometric Shapes'
    },
    'shades': {
        'chars': " ░▒▓█",
        'name': 'Shaded Blocks'
    },
    'shades_alt': {
        'chars': " ░░░▒▒▒▓▓▓███",
        'name': 'Shaded Blocks Alternative'
    },
    'shades_mix': {
        'chars': " .░▒▓█",
        'name': 'Mixed Shaded Blocks'
    },
}

DEFAULT_CHAR_SET = 'standard'