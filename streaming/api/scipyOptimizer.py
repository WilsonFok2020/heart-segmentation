from scipy.optimize._differentialevolution import DifferentialEvolutionSolver
import numpy as np
from numpy import array, asarray, float64, zeros
from scipy.optimize import OptimizeResult, minimize, _lbfgsb, LbfgsInvHessProduct
from scipy.optimize._optimize import MemoizeJac, _status_message, _check_unknown_options, _prepare_scalar_function
from scipy.optimize._constraints import old_bound_to_new
import warnings
import time

_MACHEPS = np.finfo(float64).eps

def rosenbrock_func(a,b,x):
    """
    x is a 1-D vector (N,)
    """
    return (a-x[0])**2.0 + b*((x[1]-x[0]**2.0)**2.0)





# https://github.com/scipy/scipy/blob/v1.10.0/scipy/optimize/_differentialevolution.py#L22-L399
class ModifiedEvolution(DifferentialEvolutionSolver):
    def __init__(self, func, bounds, args=(),
                        strategy='best1bin',
                        maxiter=1000, popsize=15, tol=0.01,
                        mutation=(0.5, 1), recombination=0.7, seed=None,
                        callback=None, disp=False, polish=True,
                        init='latinhypercube', atol=0, updating='immediate',
                        workers=1, constraints=(), x0=None, *,
                        integrality=None, vectorized=False):

        super().__init__(func, bounds, args=args,
                                     strategy=strategy,
                                     maxiter=maxiter,
                                     popsize=popsize, tol=tol,
                                     mutation=mutation,
                                     recombination=recombination,
                                     seed=seed, polish=polish,
                                     callback=callback,
                                     disp=disp, init=init, atol=atol,
                                     updating=updating,
                                     workers=workers,
                                     constraints=constraints,
                                     x0=x0,
                                     integrality=integrality,
                                     vectorized=vectorized)
        
        # placeholder
        self.warning_flag = False
        self.status_message = None 
        self.last_nit = None

        
    def initialize_population(self):

        self.status_message = _status_message['success']

        # The population may have just been initialized (all entries are
        # np.inf). If it has you have to calculate the initial energies.
        # Although this is also done in the evolve generator it's possible
        # that someone can set maxiter=0, at which point we still want the
        # initial energies to be calculated (the following loop isn't run).
        if np.all(np.isinf(self.population_energies)):
            self.feasible, self.constraint_violation = (
                self._calculate_population_feasibilities(self.population))

            # only work out population energies for feasible solutions
            self.population_energies[self.feasible] = (
                self._calculate_population_energies(
                    self.population[self.feasible]))

            self._promote_lowest_energy()

    def evolving(self):
        # do the optimization.
        for nit in range(1, self.maxiter + 1):
            # evolve the population by a generation
            try:
                next(self) # I am a generator!
            except StopIteration:
                self.warning_flag = True
                if self._nfev > self.maxfun:
                    self.status_message = _status_message['maxfev']
                elif self._nfev == self.maxfun:
                    self.status_message = ('Maximum number of function evaluations'
                                      ' has been reached.')
                break

            if self.disp:
                print(f"differential_evolution step {nit}: f({self.x})= {self.population_energies[0]}")

            if self.callback:
                c = self.tol / (self.convergence + _MACHEPS)
                self.warning_flag = bool(self.callback(self.x, convergence=c))
                if self.warning_flag:
                    self.status_message = ('callback function requested stop early'
                                      ' by returning True')

            # should the solver terminate?
            if self.warning_flag or self.converged():
                break

            yield (nit, self.x, self.population_energies[0])
            # slow down for visualization
            # time.sleep(1)

        else:
            self.status_message = _status_message['maxiter']
            self.warning_flag = True

        self.last_nit = nit
        print ('reaching the end.................')


    def package_results(self):

        DE_result = OptimizeResult(
            x=self.x,
            fun=self.population_energies[0],
            nfev=self._nfev,
            nit=self.last_nit,
            message=self.status_message,
            success=(self.warning_flag is not True))
        
        return DE_result
    
    def perform_polishing(self):

        DE_result = self.package_results()
        if self.polish and not np.all(self.integrality):
            DE_result = self.polishing(DE_result)
        
        return DE_result
        

    
    def polishing(self, DE_result, polish_method = 'L-BFGS-B'):
        # can't polish if all the parameters are integers
        if np.any(self.integrality):
            # set the lower/upper bounds equal so that any integrality
            # constraints work.
            limits, integrality = self.limits, self.integrality
            limits[0, integrality] = DE_result.x[integrality]
            limits[1, integrality] = DE_result.x[integrality]

        if self._wrapped_constraints:
            polish_method = 'trust-constr'

            constr_violation = self._constraint_violation_fn(DE_result.x)
            if np.any(constr_violation > 0.):
                warnings.warn("differential evolution didn't find a"
                                " solution satisfying the constraints,"
                                " attempting to polish from the least"
                                " infeasible solution", UserWarning)
        if self.disp:
            print(f"Polishing solution with '{polish_method}'")
        result = minimize(self.func,
                            np.copy(DE_result.x),
                            method=polish_method,
                            bounds=self.limits.T,
                            constraints=self.constraints)

        self._nfev += result.nfev
        DE_result.nfev = self._nfev

        # Polishing solution is only accepted if there is an improvement in
        # cost function, the polishing was successful and the solution lies
        # within the bounds.
        if (result.fun < DE_result.fun and
                result.success and
                np.all(result.x <= self.limits[1]) and
                np.all(self.limits[0] <= result.x)):
            DE_result.fun = result.fun
            DE_result.x = result.x
            DE_result.jac = result.jac
            # to keep internal state consistent
            self.population_energies[0] = result.fun
            self.population[0] = self._unscale_parameters(result.x)

        if self._wrapped_constraints:
            DE_result.constr = [c.violation(DE_result.x) for
                                c in self._wrapped_constraints]
            DE_result.constr_violation = np.max(
                np.concatenate(DE_result.constr))
            DE_result.maxcv = DE_result.constr_violation
            if DE_result.maxcv > 0:
                # if the result is infeasible then success must be False
                DE_result.success = False
                DE_result.message = ("The solution does not satisfy the "
                                        f"constraints, MAXCV = {DE_result.maxcv}")
                
        return DE_result

# from https://github.com/scipy/scipy/blob/dde50595862a4f9cede24b5d1c86935c30f1f88a/scipy/optimize/_lbfgsb_py.py#L386
class ModifiedLBFGSB:
    def __init__(self, fun, x0, args=(), jac=None, bounds=None,
                     disp=None, maxcor=10, ftol=2.2204460492503131e-09,
                     gtol=1e-5, eps=1e-8, maxfun=15000, maxiter=15000,
                     iprint=-1, callback=None, maxls=20,
                     finite_diff_rel_step=None, **unknown_options):


        
        _check_unknown_options(unknown_options)
        self.m = maxcor
        self.pgtol = gtol
        self.factr = ftol / np.finfo(float).eps
        self.callback = callback
        self.maxfun = maxfun
        self.maxiter = maxiter

        x0 = asarray(x0).ravel()
        self.n, = x0.shape

        if bounds is None:
            bounds = [(None, None)] * self.n
        if len(bounds) != self.n:
            print (bounds, self.n, x0)
            raise ValueError('length of x0 != length of bounds')

        # unbounded variables must use None, not +-inf, for optimizer to work properly
        bounds = [(None if l == -np.inf else l, None if u == np.inf else u) for l, u in bounds]
        # LBFGSB is sent 'old-style' bounds, 'new-style' bounds are required by
        # approx_derivative and ScalarFunction
        new_bounds = old_bound_to_new(bounds)

        # check bounds
        if (new_bounds[0] > new_bounds[1]).any():
            raise ValueError("LBFGSB - one of the lower bounds is greater than an upper bound.")

        # initial vector must lie within the bounds. Otherwise ScalarFunction and
        # approx_derivative will cause problems
        x0 = np.clip(x0, new_bounds[0], new_bounds[1])

        self.iprint = True

        sf = _prepare_scalar_function(fun, x0, jac=jac, args=args, epsilon=eps,
                                    bounds=new_bounds,
                                    finite_diff_rel_step=finite_diff_rel_step)

        self.func_and_grad = sf.fun_and_grad
        self.sf = sf

        fortran_int = _lbfgsb.types.intvar.dtype

        self.nbd = zeros(self.n, fortran_int)
        self.low_bnd = zeros(self.n, float64)
        self.upper_bnd = zeros(self.n, float64)
        self.bounds_map = {(None, None): 0,
                    (1, None): 1,
                    (1, 1): 2,
                    (None, 1): 3}
        for i in range(0, self.n):
            l, u = bounds[i]
            if l is not None:
                self.low_bnd[i] = l
                l = 1
            if u is not None:
                self.upper_bnd[i] = u
                u = 1
            self.nbd[i] = self.bounds_map[l, u]

        if not maxls > 0:
            raise ValueError('maxls must be positive.')
        else:
            self.maxls = maxls

        self.x = array(x0, float64)
        self.f = array(0.0, float64)
        self.g = zeros((self.n,), float64)
        self.wa = zeros(2*self.m*self.n + 5*self.n + 11*self.m*self.m + 8*self.m, float64)
        self.iwa = zeros(3*self.n, fortran_int)
        self.task = zeros(1, 'S60')
        self.csave = zeros(1, 'S60')
        self.lsave = zeros(4, fortran_int)
        self.isave = zeros(44, fortran_int)
        self.dsave = zeros(29, float64)

        self.task[:] = 'START'

        self.n_iterations = 0

    def solving(self):

        while 1:
            # x, f, g, wa, iwa, task, csave, lsave, isave, dsave = \
            _lbfgsb.setulb(self.m, self.x, self.low_bnd, self.upper_bnd, self.nbd, self.f, self.g, self.factr,
                        self.pgtol, self.wa, self.iwa, self.task, self.iprint, self.csave, self.lsave,
                        self.isave, self.dsave, self.maxls)
            task_str = self.task.tobytes()
            if task_str.startswith(b'FG'):
                # The minimization routine wants f and g at the current x.
                # Note that interruptions due to maxfun are postponed
                # until the completion of the current minimization iteration.
                # Overwrite f and g:
                self.f, self.g = self.func_and_grad(self.x)
                yield (self.n_iterations, self.x, self.sf.nfev,self.f,self.g)

            elif task_str.startswith(b'NEW_X'):
                
                # new iteration
                self.n_iterations += 1
                if self.callback is not None:
                    self.callback(np.copy(self.x))

                if self.n_iterations >= self.maxiter:
                    self.task[:] = 'STOP: TOTAL NO. of ITERATIONS REACHED LIMIT'
                elif self.sf.nfev > self.maxfun:
                    self.task[:] = ('STOP: TOTAL NO. of f AND g EVALUATIONS '
                            'EXCEEDS LIMIT')
                # this step calculates the solution at the end of an iteration
                # thus, we cannot see the steps taken to evaluate numerical gradient at f'(x)
                # yield (self.n_iterations, self.x, self.sf.nfev,self.f,self.g)
            else:
                break

            # print ('self.n_iterations {}, self.x {}, self.sf.nfev {}, f {}, g {}'.format(
            #     self.n_iterations,
            #     self.x,
            #     self.sf.nfev,
            #         self.f,
            #         self.g))
            


    def package_result(self):

        task_str = self.task.tobytes().strip(b'\x00').strip()
        if task_str.startswith(b'CONV'):
            warnflag = 0
        elif self.sf.nfev > self.maxfun or self.n_iterations >= self.maxiter:
            warnflag = 1
        else:
            warnflag = 2

        # These two portions of the workspace are described in the mainlb
        # subroutine in lbfgsb.f. See line 363.
        s = self.wa[0: self.m*self.n].reshape(self.m, self.n)
        y = self.wa[self.m*self.n: 2*self.m*self.n].reshape(self.m, self.n)

        # See lbfgsb.f line 160 for this portion of the workspace.
        # isave(31) = the total number of BFGS updates prior the current iteration;
        n_bfgs_updates = self.isave[30]

        n_corrs = min(n_bfgs_updates, self.m)
        hess_inv = LbfgsInvHessProduct(s[:n_corrs], y[:n_corrs])

        task_str = task_str.decode()
        result = OptimizeResult(fun=self.f, jac=self.g, nfev=self.sf.nfev,
                            njev=self.sf.ngev,
                            nit=self.n_iterations, status=warnflag, message=task_str,
                            x=self.x, success=(warnflag == 0), hess_inv=hess_inv)
        
        print (result)
        return result
