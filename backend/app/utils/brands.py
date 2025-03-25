brands = [
    "a. le coq", "agronom", "air wick", "alma", "almeda", "almeda nature care",
    "almeda spa", "alpro", "always", "ariel", "aura", "banquet", "bolsius",
    "bonduelle", "bref", "bullet", "clerit", "colgate", "cozy home", "decorata party",
    "dove", "dr.oetker", "eesti pagar", "elseve", "erich krause", "estrella",
    "evoluderm", "fa", "farmi", "fazer", "felix", "frosch", "garnier", "gillette",
    "golden lady", "good cook", "hipp", "huggies", "i love eco", "ica", "jacobs",
    "jordan", "kalev", "l'oreal paris", "lay's", "lego", "lenor", "libresse",
    "lucia elite", "maggi", "maks & moorits", "maku", "mamma", "marmiton",
    "maybelline new york", "mayeri", "milka", "mywear", "määramata", "nescafe",
    "nestle", "nivea", "nopri", "nurme", "nutribalance", "nõo", "old spice",
    "organic way", "oskar", "osram", "palmolive", "pampers", "pantene", "persil",
    "pompea", "prosperplast", "puhas loodus", "põltsamaa", "põnn", "rakvere",
    "rannarootsi", "rexona", "rimi", "rimi planet", "rimi smart", "saaremaa",
    "saku", "salvest", "santa maria", "selection by rimi", "silan", "syoss",
    "tallegg", "tartu mill", "tere", "tri-bio", "ty", "vici", "wild & mild", "zewa",
    "12e mezzo", "alma", "alpro", "amalattea", "aptamil", "aramis", "axa",
    "bacardi", "bahlsen", "balconi", "ballantines", "baltika", "barni", "belvita",
    "blockbuster", "brunberg", "captain morgan", "casa charlize", "casino", "corny",
    "daim", "deary", "doppio passo", "emma", "epicuro", "e-piim", "esfolio", "farm milk",
    "farmi", "fazer", "fazer geisha", "ferrero", "friso", "garling", "geisha",
    "grafia plus", "havana club", "henske", "hipp", "holle", "imperial xii",
    "jamie parra", "jose cuervo", "kalev", "karl fazer", "karuna", "kex", "kinder",
    "kit kat", "kupiec", "leader", "lindt", "loctite", "lotte", "m&m's", "magnum",
    "marabou", "metaxa", "milka", "milka", "milky sip", "milupa", "mio delizzi",
    "mio&rio", "mo saaremaa", "moment", "monarque", "nescafe", "nestle", "nopri",
    "nuppi", "nurr", "oakheart", "oatly", "oreo", "papale", "pergale", "piimameister otto",
    "pipra naps", "piprasnaps", "põnn", "riga balsam", "ritter sport", "rooster rojo",
    "roshen", "saare", "salvest", "schogeten", "semper", "sessantanni", "sierra",
    "skrivaru", "start", "steenland", "stroh", "tartu mill", "tere", "toblerone",
    "tres sombreros", "tutteli", "väike tom", "valio", "valio gefilus", "valio profeel",
    "vilma", "vita+", "well done", "xpoint", "zott"
]

def getBrand(title):
    for brand in brands:
        if brand in title:
            return brand
    return None

# Check if both products have the same brand, returns stripped strings or None
def compareBrands(title1, title2):
    brand1 = ''
    brand2 = ''
    for brand in brands:
        if brand in title1:
            brand1 = brand
        if brand in title2:
            brand2 = brand

    if brand1 is not brand2:
        return None
    else:
        return [title1.replace(brand1, ''), title2.replace(brand2, '')]
