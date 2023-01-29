from flask import current_app, render_template, stream_template, Response, request
from .scipyOptimizer import ModifiedEvolution, rosenbrock_func, ModifiedLBFGSB
import time, os
import numpy as np
from functools import partial
from api import app

def extract_form(request):
    # we cannot get json as we did not submit json >_<
    # variable_name = request.get_json()
    # 

    a = request.form.get('a')
    b = request.form.get('b')
    lower = request.form.get('lower')
    upper = request.form.get('upper')
    maxiter = request.form.get('maxiter')

    initialGuess = request.form.get('initialGuess')
    print ('initialGuess ', initialGuess)
    initialGuess = [float(value) for value in initialGuess.split(',')]

    # values in request objects are string
    func = partial(rosenbrock_func, float(a), float(b))
    maxiter = int(maxiter)
    lower = float(lower)
    upper = float(upper)

    bounds = [(lower,upper), (lower,upper)]
    print ('bounds ', bounds)

    return bounds, maxiter, initialGuess, func, a, b, lower, upper
    

@app.route('/')
def hello_world():
    return render_template("callback.html", maxiter=current_app.maxiter,
                            a=current_app.a,
                              b=current_app.b,
                                lower=current_app.lower,
                                  upper=current_app.upper,
                                    initialGuess=current_app.initialGuess)


@app.route("/solver", methods=['POST'])
def solver():
    
    methods = request.form.get('methods')
    print ('methods = ', methods)
    if methods == "Differential Evolution":
        return callEvolution()
    elif methods == "LBFGS":
        return call_LBFGS()
    else:
        raise RuntimeError('unknown methods')

@app.route("/evolve", methods=["POST"])
def callEvolution():
     
    bounds, maxiter, initialGuess, func,  a, b, lower, upper = extract_form(request)

    # start a timer to see how long it takes to solve the problem
    start_time = time.time()

    def add_html():
        """
        Create a generator that produces the intermediate results of the optimizer
        """
        # using a context manager means that any created Pool objects are
        # cleared up.
        # don't pass args because they try to enter the f(x) as arguments
        with ModifiedEvolution(func, bounds, maxiter=maxiter,
                             polish=True, disp=True) as solver:
            
            solver.initialize_population()
            for intermediate_results in solver.evolving():
                yield (intermediate_results, time.time()-start_time)
                
            result = solver.perform_polishing()

        yield (("final", result.x, result.fun), time.time()-start_time)
            

    return Response(stream_template('callback.html',
                                     results=add_html(),
                                    initialGuess=','.join([str(i) for i in initialGuess]),
                                    maxiter=maxiter, a=a, b=b, 
                                    lower=lower, upper=upper))

@app.route('/lbfgs', methods=['POST'])
def call_LBFGS():
    
    bounds, maxiter, initialGuess, func,  a, b, lower, upper = extract_form(request)
    # start a timer to see how long it takes to solve the problem
    start_time = time.time()

    solver = ModifiedLBFGSB(func, initialGuess, maxiter=maxiter, bounds=bounds)
    
    def add_html(solver):
        """
        Create a generator that produces the intermediate results of the optimizer
        """
        
        for intermediate_results in solver.solving():
            yield (intermediate_results, time.time()-start_time)
                
        result = solver.package_result()

        yield (("final", result.x, result.nfev, result.fun, result.jac), time.time()-start_time)
            
    return Response(stream_template('callback.html',
                                     results=add_html(solver),
                                    initialGuess=','.join([str(i) for i in initialGuess]),
                                    maxiter=maxiter, a=a, b=b, 
                                    lower=lower, upper=upper))
    