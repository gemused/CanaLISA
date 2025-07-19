import src.lisaHTI as hti

PATH_orbit_data = os.path.join(PATH_bethLISA, "dist/orbit_data/")

epoch = _choose_epoch(PATH_orbit_data + "orbits.h5", 0.5)

mass1 = # sm?
mass2 = 
mass3 = 
sma_outer = # parsec?
dist = # parsec?
data_path = 

orbit_data = hti.OrbitData(
    mass1=mass1, mass2=mass2, mass3=mass3, 
    sma_outer=args.sma_outer, dist=args.dist, 
    data_path=data_path, epoch=epoch
)

hti = hti.HierarchicalTriple(
    orbit_data, include_doppler=bool(args.include_doppler), 
    verbose=True
)

