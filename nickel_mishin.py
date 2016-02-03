import crystallography_oo as croo

nickelelastic = {'11': 247.862330908453e9,'12': 147.828379827956e9,'44': 124.838117598312e9}
nickelsurface = {'111': 1629.0e-3, '110': 2049.0e-3, '100': 1878.0e-3}
nickelunstable = 366.0e-3
nickel = croo.FCC(nickelelastic,nickelsurface,nickelunstable)
