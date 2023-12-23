import os
import sys
import shutil
import multiprocessing

from modeller import (environ, selection, alignment, model, physical,
                      group_restraints)
from modeller.automodel import automodel, refine
from modeller.optimizers import actions as actions_optimize
from modeller.optimizers import conjugate_gradients, quasi_newton
from modeller.parallel import job, local_slave
from modeller.scripts import complete_pdb


class Automodel_averaging_base(object):

    def set_defaults(self):
        self.__class__.__bases__[-1].set_defaults(self)
        # An intermediate structure will be written every 'intermediate_step'
        # MD steps.
        self.intermediate_step = 35
        self.altmod_average_prefix = "averaged_model"


    def single_model_pass(self, atmsel, num, sched):
        """Perform a single pass of model optimization"""
        actions = self.get_optimize_actions()

        for (numstep, step) in enumerate(sched):
            molpdf = step.optimize(atmsel, output=self.optimize_output,
                                   max_iterations=self.max_var_iterations,
                                   actions=actions)
            self.write_int(numstep + 1, num)
            # Also check for molpdf being NaN (depends on Python version; on 2.3
            # x86 it evaluates as equal to everything; with 2.4 x86 it is
            # not greater or smaller than anything)
            if molpdf > self.max_molpdf \
               or (molpdf == 0. and molpdf == 1.) \
               or (not molpdf >= 0. and not molpdf < 0):
                log.error('single_model',
                          "Obj. func. (%.3f) exceeded max_molpdf (%.3f) " \
                                      % (molpdf, self.max_molpdf))
        actions = self.get_refine_actions()
        #######################################################################
        # at T=1000, max_atom_shift for 4fs is cca 0.15 A.
        current_output_dirpath = "%s_intermediates_%s" % (
                                 self.altmod_average_prefix, num)

        if os.path.isdir(current_output_dirpath):
            shutil.rmtree(current_output_dirpath)
        os.mkdir(current_output_dirpath)

        interm_filepath = os.path.join(current_output_dirpath,
                                       'model.D9999%04d.pdb')
        write_interm_action = actions_optimize.write_structure(
                                  self.intermediate_step, interm_filepath)
        actions.append(write_interm_action)
        #######################################################################
        self.refine(atmsel, actions)


class Automodel_averaging(Automodel_averaging_base, automodel):
    pass


def run_averaging(automodel_obj, n_runs=4, n_jobs=1,
                  scoring_func="dope",
                  select_top=0.1,
                  cluster_cut=0.0,
                  model_name="averaged_model"):
    """
    Perform averaging in the MODELLER homology modeling process.
    # Arguments
        automodel_obj: an 'automodel' object from the 'Automodel_averaging' or
            'Automodel_statistical_potential_averaging' classes.
        n_runs: number of runs to perform (each run correspond to a single
            3D model which will be built and from which intermediate structures
            will be extracted).
        n_jobs: number of parallel jobs to use in order to build the 'n_runs'
            models.
        scoring_func: scoring function used to score the decoys. If it is a
            callable function, it must take as argument a list of filepath for
            3D structures and must return a list of scores (the lower, the
            better) for each structure. By default this is set to 'dope' and
            the 'score_with_dope' function in this module will be used.
        select_top: fraction of the decoys to take in order to perform
            averaging.
        cluster_cut: parameter to be used by the 'transfer_xyz' function of
            MODELLER while averaging the decoys.
        model_name: prefix used to name the files written when using this
            function.

    # Returns
        averaged_filepath: a filepath of the final averaged model.
    """

    if not issubclass(automodel_obj.__class__, Automodel_averaging_base):
        raise TypeError(("The automodel class of the 'automodel_obj' provided"
                         " can not be used by the 'run_averaging' function."))
    if not hasattr(scoring_func, "__call__"):
        if not scoring_func in ("dope", ):
            raise KeyError("Unknown 'scoring_func': %s" % scoring_func)
        if scoring_func == "dope":
            scoring_func = score_with_dope

    # Set up the options.
    if select_top > 1 or select_top <= 0:
        raise ValueError("Invalid 'select_top' value.")

    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    if n_jobs != 1:  # Use multiple CPUs in a parallel job on this machine.
        j = job(host='localhost')
        for n_job in range(n_jobs):
            j.append(local_slave())


    # Actually run 3D model building.
    print("# Building 3D models (for %s runs)..." % n_runs)
    automodel_obj
    automodel_obj.starting_model = 1
    automodel_obj.ending_model = n_runs
    automodel_obj.altmod_average_prefix = model_name
    if n_jobs != 1:
        automodel_obj.use_parallel_job(j)  # Use the job for model building.
    automodel_obj.make()

    # Get the output from the 3D model building phase.
    models_built = [m for m in automodel_obj.outputs if m["failure"] is None]
    if not models_built:
        raise ValueError(("No models were successfully built, can not perform"
                          " averaging."))
    decoys_filepaths = []
    for model_out in models_built:
        int_dirpath = "%s_intermediates_%s" % (model_name, model_out["num"])
        int_filepaths = [os.path.join(int_dirpath, fn) for fn \
                         in os.listdir(int_dirpath) \
                         if fn.endswith(".pdb")]
        decoys_filepaths.extend(int_filepaths)

    # Score the decoys.
    print("# Scoring %s decoys..." % len(decoys_filepaths))
    scores = scoring_func(decoys_filepaths)

    # Sort the decoys according to their score.
    decoys_info = list(zip(decoys_filepaths, scores))
    sorted_decoys_info = sorted(decoys_info, key=lambda t: t[1])
    sorted_decoys_filepaths = [t[0] for t in sorted_decoys_info]
    n_sel_decoys = int(len(sorted_decoys_filepaths)*select_top)
    if n_sel_decoys == 0:
        n_sel_decoys = 1
    print("# Selected top %s decoys:" % n_sel_decoys)
    for i in range(n_sel_decoys):
        print("(%s) %s: %s" % (i+1, sorted_decoys_info[i][0],
                               sorted_decoys_info[i][1]))

    # Average the selected decoys.
    print("# Averaging...")
    averaged_model = cluster_pdbs(sorted_decoys_filepaths[0:n_sel_decoys],
                                  cluster_cut=cluster_cut, verbose=False)
    averaged_filepath = '%s.pdb' % model_name
    averaged_model.write(file=averaged_filepath)

    # Optimize the averaged model.
    print("# Optimizing the averaged model...")
    optimize_structure(pdb_filepath=averaged_filepath,
                       output_filepath=averaged_filepath,
                       rst_filepath=automodel_obj.csrfile, verbose=False,
                       w_dope=0.5, algorithm="cg", max_iterations=200)

    print("# Completed.")

    return averaged_filepath


def score_with_dope(decoys_filepaths):
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")

    env = environ()
    env.libs.topology.read(file='$(LIB)/top_heav.lib')
    env.libs.parameters.read(file='$(LIB)/par.lib')
    dope_scores = []
    for filepath in decoys_filepaths:
        # Read a model previously generated by Modeller's automodel class.
        mdl = complete_pdb(env, filepath)
        atmsel = selection(mdl)
        score = atmsel.assess_dope()
        dope_scores.append(score)
    sys.stdout = original_stdout

    return dope_scores


def optimize_structure(pdb_filepath, output_filepath=None,
                       rst_filepath=None, verbose=False,
                       w_dope=0.5,
                       algorithm="cg", max_iterations=200,
                       suffix=".opt",
                       evaluate_en=False):

    if not algorithm in ("cg", "qn"):
        raise ValueError("Unknown optimization algorithm: %s." % algorithm)


    if not verbose:
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    env = environ()
    env.edat.dynamic_sphere = True

    env.libs.topology.read(file='$(LIB)/top_heav.lib')
    env.libs.parameters.read(file='$(LIB)/par.lib')

    # Allow calculation of statistical (dynamic_modeller) potential
    if w_dope is not None:
        env.edat.dynamic_modeller = True
        env.edat.contact_shell = 8.0
        env.edat.dynamic_sphere = True
        env.schedule_scale = physical.values(default=1.0,
                                             nonbond_spline=w_dope,)

    else:
        env.edat.dynamic_sphere = True

    mdl = complete_pdb(env, pdb_filepath)

    if w_dope is not None:
        gprsr = group_restraints(env, classes='$(LIB)/atmcls-mf.lib',
                                 parameters='$(LIB)/dist-mf.lib')
        mdl.group_restraints = gprsr

    atmsel = selection(mdl)

    if rst_filepath != None:
        mdl.restraints.append(file=rst_filepath)
    else:
        mdl.restraints.make(atmsel, restraint_type='stereo',
                            spline_on_site=False)

    if evaluate_en:
        (molpdf_init, terms_init) = atmsel.energy()

    # Prepare the optimizer.
    if algorithm == "cg":
        cg = conjugate_gradients(output='REPORT')
    elif algorithm == "qn":
        cg = quasi_newton(output='REPORT')

    # Run optimization on the all-atom selection.
    cg.optimize(atmsel, max_iterations=max_iterations)
    (molpdf, terms) = atmsel.energy()


    if evaluate_en:
        print("\n# Initial energy:")
        print(molpdf_init, terms_init)

    print("\n# Final energy:")
    print(molpdf, terms)


    if output_filepath is None:
        output_filepath = pdb_filepath + suffix

    mdl.write(file=output_filepath)

    if not verbose:
        sys.stdout = original_stdout

    return molpdf, terms


def cluster_pdbs(pdb_filepaths, cluster_cut=1.5, verbose=False):
    """
    Get a representative of a set of PDBs with the same sequence.
    A representative model is returned.
    Adapted from: https://salilab.org/modeller/wiki/Cluster%20PDBs
    """

    if not verbose:
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    env = environ()
    ali = alignment(env)

    # Read all structures, and make a 1:1 alignment of their sequences
    for i, pdb_filepath in enumerate(pdb_filepaths):
        print("- Parsing model %s." % i)
        m = model(env, file=pdb_filepath)
        ali.append_model(m, align_codes="%s_model" % i, atom_files=pdb_filepath)

    # Structurally superimpose all structures without changing the alignment
    ali.malign3d(gap_penalties_3d=(0, 3), fit=False)

    # Add a new dummy model with the same sequence to hold the cluster. This
    # represents the target sequence in the alignment, while the previous
    # sequences represent the templates.
    m = model(env, file=pdb_filepaths[0])
    ali.append_model(m, align_codes='cluster', atom_files='cluster.opt')

    # Make the clustered representative
    m.transfer_xyz(ali, cluster_cut=cluster_cut)

    if not verbose:
        sys.stdout = original_stdout

    return m
