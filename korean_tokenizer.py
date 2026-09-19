import re
_JOSA = ["으로부터","에서부터","한테서","로부터","에게서","이라고","라고","으로서","로서","에서","에게","한테","에서도","에게도","으로도","로도","까지도","부터도","이라도","라도","이라면","라면","으로는","로는","에서는","에게는","이지만","지만","이어서","여서","이어도","여도","이어야","여야","으로","로","와","과","이랑","랑","부터","까지","마다","밖에","처럼","만큼","보다","이나","나","이든","든","이며","며","에도","에만","에는","에서만","에서도","이라","라","이고","고","이가","가","이","을","를","은","는","의","도","만","에"]
_EOMI = ["했습니다","합니다","됩니다","있습니다","없습니다","입니다","십니다","겠습니다","했어요","해요","돼요","있어요","없어요","하는","하고","해서","하면","하지","하여","되는","되고","돼서","되면","되지","되어","있는","있고","있어서","있으면","없는","없고","없어서","없으면","한다","된다","있다","없다","하였다","되었다","였다","이었다","하며","되며","이며","했다","됐다","할","될","있을","없을","하기","되기","있기","없기","함","됨","있음","없음","해야","돼야","있어야","없어야","하도록","되도록","하여도","되어도","이다","이고","이며","이어서","이어도","이어야","이지만","이라도","이었다","이었고","이었으며"]
_JOSA_SORTED=sorted(_JOSA,key=len,reverse=True);_EOMI_SORTED=sorted(_EOMI,key=len,reverse=True)
_HANGUL_RE=re.compile(r"[가-힣]+");_PUNCT_RE=re.compile(r"[^\w가-힣]")
def _strip_suffix(word,sl):
    for s in sl:
        if word.endswith(s) and len(word)>len(s)+1: return word[:-len(s)]
    return word
def normalize_token(token):
    token=_PUNCT_RE.sub("",token)
    if not token: return ""
    if re.match(r'^\d+([.,]\d+)?$',token): return 'NUM'
    if not _HANGUL_RE.search(token): return token.lower()
    stem=_strip_suffix(token,_JOSA_SORTED);stem=_strip_suffix(stem,_EOMI_SORTED)
    return stem if stem else token
def tokenize(text):
    return [n for n in (normalize_token(r) for r in text.strip().split()) if n]
def tokenize_dual(text):
    pairs=[]
    for raw in text.strip().split():
        n=normalize_token(raw)
        if n: pairs.append((raw,n))
    return pairs
