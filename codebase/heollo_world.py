# coding=utf-8

def myCoolfunction(myCoolmessage: str, show: bool = False) -> str:
    """
    Produit un 'message de bienvenue' sur mesure et normaliser.

    :param myCoolmessage: Le message à ajouter
    :param show: Option pour imprimer le message dans la console
    :return: Le message
    """
    msg = f"hellow world and {myCoolmessage}"
    if show:
        print(msg)
    return msg


def hhltest():
    pass
    return None
