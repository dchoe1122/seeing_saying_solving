import spot

def test_containment():
    f1 = spot.formula("G(a -> Fb)")
    f2 = spot.formula("G(a -> F(b | c))")
    assert spot.contains(f2, f1)
    assert not spot.contains(f1, f2)