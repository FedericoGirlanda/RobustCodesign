from jinja2 import Environment, FileSystemLoader

def generateUrdf(Mp,lp,Jp, name = None): 
    # Compile templates into URDF robot description
    loader = FileSystemLoader(searchpath="data/cart_pole/urdfs/template")
    env = Environment(loader=loader, autoescape=True)

    dir = "data/cart_pole/urdfs/"
    if name is None:
        t = dir + "cartpole_CMAES.urdf"
    else:
        t = dir + name
    template = env.get_template("cartpoleTemplate.j2")
    f = open(t, "w")
    f.write(template.render(Mp = Mp, 
                            lp = lp,
                            Jp = Jp))
    
    return t