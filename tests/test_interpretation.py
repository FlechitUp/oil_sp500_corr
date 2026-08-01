from src.interpretation import classify_episode

def test_categories():
    assert classify_episode(-.08,-.01,-.01,-.1).startswith("Caída principalmente")
    assert "USD/BRL" in classify_episode(-.01,-.08,-.01,-.1)
    assert classify_episode(-.04,-.04,-.01,-.09)=="Caída conjunta"
    assert classify_episode(-.05,.03,-.01,-.03).startswith("Efectos opuestos")
    assert classify_episode(-.01,-.01,-.08,-.1).startswith("Movimiento predominantemente")
    assert classify_episode(0,0,0,-.005)=="Sin drawdown relevante"

