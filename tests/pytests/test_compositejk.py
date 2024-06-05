import pytest

from utils import compare_values, compare
from addons import using

import psi4

pytestmark = [pytest.mark.psi, pytest.mark.api]

@pytest.fixture
def mols():
    return {
        "h2o" : psi4.geometry("""
0 1
O    0.000000000000     0.000000000000    -0.124038860300
H    0.000000000000    -1.431430901356     0.984293362719
H    0.000000000000     1.431430901356     0.984293362719
units au
"""),
        "nh2" : psi4.geometry("""
0 2
N    0.000000000000000   0.000000000000000  -0.145912918634892
H    0.000000000000000  -1.511214298139000   1.013682596946108
H    0.000000000000000   1.511214298139000   1.013682596946108
units au
"""),
        "h2o_nap1" : psi4.geometry("""
0 1
O    0.000000000000     0.000000000000    -0.124038860300
H    0.000000000000    -1.431430901356     0.984293362719
H    0.000000000000     1.431430901356     0.984293362719
--
1 1
Na   0.000000000000     0.000000000000    -4.124038860300
units au
""")
        }

@pytest.mark.smoke
@pytest.mark.parametrize("j_algo", [ 
        pytest.param("DFDIRJ") 
    ]
) #to be extended in the future
@pytest.mark.parametrize("k_algo", [ 
        pytest.param("LINK"),
        pytest.param("COSX"),
        pytest.param("SNLINK", marks=using('gauxc')),
    ]
) #to be extended in the future
def test_composite_call(j_algo, k_algo, mols, request):
    """Test all SCF_TYPE={J}+{K} combinations for an HF calculation.
    The correct algorithm pair should be called in each case.""" 

    # initial option setup 
    test_id = request.node.callspec.id
    
    molecule = mols["h2o"]
    
    scf_type = f'{j_algo}+{k_algo}'
    psi4.set_options({ "scf_type" : f'{j_algo}+{k_algo}', "incfock": True, "save_jk": True })
  
    if any([ _ in scf_type for _ in ["COSX", "SNLINK"] ]): 
        psi4.set_options({ "screening" : "schwarz" })
    else:
        psi4.set_options({ "screening" : "density" })

    # run composite JK algorithm
    E, wfn = psi4.energy("hf/6-31g", molecule=molecule, return_wfn=True) 

    clean_j_name, clean_k_name = wfn.jk().name().split("+")

    # check that correct J algo has been called
    clean_j_name = clean_j_name.replace("-", "") # replace DF-DirJ with DFDirJ
    assert clean_j_name.lower() == j_algo.lower(), f'{test_id} has correct J build method'

    # check that correct K algo has been called
    clean_k_name = clean_k_name.replace("-", "") # replace sn-LinK with snLinK
    assert clean_k_name.lower() == k_algo.lower(), f'{test_id} has correct K build method'

@pytest.mark.parametrize(
    "inp",
    [
        pytest.param({"method" : "hf",
                      "options": {"reference" : "rhf"},
                      "molecule" : "h2o",
                      "bsse_type" : None,
                      },
                      marks=pytest.mark.quick,
                      id="h2o (rhf)"),
        pytest.param({"method" : "b3lyp",
                      "options": {"reference" : "rhf"},
                      "molecule" : "h2o",
                      "bsse_type" : None,
                      },
                      marks=pytest.mark.quick,
                      id="h2o (rks)"),
        pytest.param({"method" : "hf",
                      "options": {"reference" : "uhf"},
                      "molecule" : "nh2",
                      "bsse_type" : None,
                      },
                      id="nh2 (uhf)"),
        pytest.param({"method" : "hf",
                      "options": {"reference" : "rohf"},
                      "molecule" : "nh2",
                      "bsse_type" : None,
                      },
                      id="nh2 (rohf)"),
        pytest.param({"method" : "hf",
                      "options": {"reference" : "rhf"},
                      "molecule" : "h2o_nap1",
                      "bsse_type" : "CP",
                      },
                      marks=pytest.mark.nbody,
                      id="h2o/na+ (rhf ie)"),
    ],
)
@pytest.mark.parametrize(
    "scf",
    [
        pytest.param({"scf_type" : "dfdirj+cosx",
                      "ref" : { 
                          "h2o (rhf)" : -76.026780223322,
                          "h2o (rks)" : -76.420402720419,
                          "nh2 (uhf)" : -55.566890252551,
                          "nh2 (rohf)" : -55.562689948780,
                          "h2o/na+ (rhf ie)" : -0.040121884077,
                      },
                      },
                      id="cosx"),
        pytest.param({"scf_type" : "dfdirj+snlink",
                      "snlink_force_cartesian": False,
                      "ref" : { 
                          "h2o (rhf)" : -76.026788692185, 
                          "h2o (rks)" : -76.420403557357,
                          "nh2 (uhf)" : -55.566911357539,
                          "nh2 (rohf)" : -55.562710424257,
                          "h2o/na+ (rhf ie)" : -0.040118757043,
                      },
                      },
                      id="snlink (spherical)", marks=using("gauxc")),
        pytest.param({"scf_type" : "dfdirj+snlink",
                      "snlink_force_cartesian": True,
                      "ref" : { 
                          "h2o (rhf)" : -76.026788692185, 
                          "h2o (rks)" : -76.420403557357,
                          "nh2 (uhf)" : -55.566911357539,
                          "nh2 (rohf)" : -55.562710424257,
                          "h2o/na+ (rhf ie)" : -0.040118757043,
                      },
                      },
                      id="snlink (cartesian)", marks=using("gauxc")),
 
    ]
)
def test_seminum(inp, scf, mols, request):
    """Test the DF-DirJ + COSX/sn-LinK JK objects via SCF calculations"""

    test_id = request.node.callspec.id
    
    molecule = mols[inp["molecule"]]
    psi4.set_options({"scf_type" : scf["scf_type"], "basis": "cc-pvdz"})
    psi4.set_options(inp["options"])
    if "snlink_force_cartesian" in scf.keys():
        psi4.set_options({"snlink_force_cartesian": scf["snlink_force_cartesian"]})

        #SNLINK_FORCE_CARTESIAN doesnt work with symmetry currently
        molecule.reset_point_group("C1")

    # does the SCF energy match a pre-computed reference?
    energy_seminum = psi4.energy(inp["method"], molecule=molecule, bsse_type=inp["bsse_type"])
    assert compare_values(scf["ref"][test_id.split("-")[1]], energy_seminum, 6, f'{test_id} accurate to reference (1e-6 threshold)')

    # is the SCF energy reasonably close to a conventional SCF?
    psi4.set_options({"scf_type" : "pk"})
    energy_pk = psi4.energy(inp["method"], molecule=molecule, bsse_type=inp["bsse_type"])
    assert compare_values(energy_pk, energy_seminum, 4, f'{test_id} DFDIRJ+COSX accurate to PK (1e-4 threshold)')

@pytest.mark.parametrize(
    "inp",
    [
        pytest.param({"method" : "hf",
                      "options": {"reference" : "rhf"},
                      "molecule" : "h2o",
                      "bsse_type" : None,
                      },
                      id="h2o (rhf)"),
        pytest.param({"method" : "b3lyp",
                      "options": {"reference" : "rhf"},
                      "molecule" : "h2o",
                      "bsse_type" : None,
                      },
                      id="h2o (rks)"),
        pytest.param({"method" : "hf",
                      "options": {"reference" : "uhf"},
                      "molecule" : "nh2",
                      "bsse_type" : None,
                      },
                      id="nh2 (uhf)"),
        pytest.param({"method" : "hf",
                      "options": {"reference" : "rohf"},
                      "molecule" : "nh2",
                      "bsse_type" : None,
                      },
                      id="nh2 (rohf)"),
        pytest.param({"method" : "hf",
                      "options": {"reference" : "rhf"},
                      "molecule" : "h2o_nap1",
                      "bsse_type" : "CP",
                      },
                      marks=pytest.mark.nbody,
                      id="h2o/na+ (rhf ie)"),
    ],
)
@pytest.mark.parametrize(
    "scf",
    [
        pytest.param({"scf_type" : "dfdirj+cosx"},
                      id="cosx"),
        pytest.param({"scf_type" : "dfdirj+snlink"},
                      id="snlink", marks=using("gauxc")),
    ]
)
def test_seminum_incfock(inp, scf, mols, request):
    """Test the efficiency of IncFock in DF-DirJ + COSX/sn-LinK JK objects via SCF calculations"""

    test_id = request.node.callspec.id

    molecule = mols[inp["molecule"]]
    psi4.set_options({"scf_type" : scf["scf_type"], "basis": "cc-pvdz", "incfock": False})
    psi4.set_options(inp["options"])

    # compute energy+wfn without IncFock 
    energy_seminum_noinc, wfn_seminum_noinc = psi4.energy(inp["method"], molecule=molecule, bsse_type=inp["bsse_type"], return_wfn=True)
    #assert compare_values(inp["ref"], energy_dfjcosk, atol=1e-6)

    # compute energy+wfn with Incfock 
    psi4.set_options({"incfock" : True})
    energy_seminum_inc, wfn_seminum_inc = psi4.energy(inp["method"], molecule=molecule, bsse_type=inp["bsse_type"], return_wfn=True)

    # how do energies compare?
    assert compare_values(energy_seminum_noinc, energy_seminum_inc, 6, f'{test_id} IncFock accurate (1e-6 threshold)')
    
    # how do SCF iteration counts compare?
    niter_noinc = int(wfn_seminum_noinc.variable("SCF ITERATIONS")) if wfn_seminum_noinc.has_scalar_variable("SCF ITERATIONS") else 0
    niter_inc = int(wfn_seminum_inc.variable("SCF ITERATIONS")) if wfn_seminum_inc.has_scalar_variable("SCF ITERATIONS") else 0
    
    assert compare(True, abs(niter_inc - niter_noinc) <= 3, f'{test_id} IncFock efficient')

@pytest.mark.parametrize("functional", [ "bp86", "b3lyp" ])
@pytest.mark.parametrize(
    "scf_type", 
    [ 
        pytest.param("DFDIRJ"), 
        pytest.param("LINK"),
        pytest.param("COSX"),
        pytest.param("SNLINK", marks=using('gauxc')),
        pytest.param("DFDIRJ+LINK"),
        pytest.param("DFDIRJ+COSX"),
        pytest.param("DFDIRJ+SNLINK", marks=using('gauxc')),
    ]
)
def test_dfdirj(functional, scf_type, mols):
    """Test the functionality of the SCF_TYPE keyword for CompositeJK methods under varying situations:
      - Using hybrid DFT functionals without specifying a K algorithm should cause a RuntimeError to be thrown.
      - Not specifying a J algorithm should cause a ValidationError to be thrown."""

    composite_algo_to_matrix = {
        "DFDIRJ": "J",
        "LINK" : "K",
        "COSX": "K",
        "SNLINK": "K",
    }

    molecule = mols["h2o"]
    screening = "CSAM" if any([ _ in scf_type for _ in [ "COSX", "SNLINK" ] ]) else "DENSITY"
    
    # if J algorithm isn't specified, code should throw here...
    if not any([ algo in scf_type for algo, matrix in composite_algo_to_matrix.items() if matrix == "J" ]):
        with pytest.raises(psi4.ValidationError) as e_info:
            psi4.set_options({"scf_type": scf_type, "reference": "rhf", "basis": "cc-pvdz", "screening": screening}) 

        # we keep this line just for printout purposes; should always pass if done correctly 
        assert compare(type(e_info), pytest.ExceptionInfo, f'{scf_type}+{functional} throws ValidationError')
 
    # ... else we continue as normal 
    else:  
        psi4.set_options({"scf_type": scf_type, "reference": "rhf", "basis": "cc-pvdz", "screening": screening}) 
    
        is_hybrid = True if functional == "b3lyp" else False
        k_algo_specified = True if any([ algo in scf_type for algo, matrix in composite_algo_to_matrix.items() if matrix == "K" ]) else False

        # if K algorithm isn't specified, but hybrid functional is used, code should throw...
        if is_hybrid and not k_algo_specified: 
            with pytest.raises(RuntimeError) as e_info:
                E = psi4.energy(functional, molecule=molecule) 

            # we keep this line just for printout purposes; should always pass if done correctly 
            assert compare(type(e_info), pytest.ExceptionInfo, f'{scf_type}+{functional} throws RuntimeError')
    
        # ... else code will run fine
        else:
            E = psi4.energy(functional, molecule=molecule) 

            # we keep this line just for printout purposes; should always pass if done correctly 
            assert compare(type(E), float, f'{scf_type}+{functional} executes')

@pytest.mark.parametrize("j_algo", [ "DFDIRJ" ]) #to be extended in the future
@pytest.mark.parametrize(
    "k_algo", 
    [ 
        pytest.param("LINK"),
        pytest.param("COSX"),
        pytest.param("SNLINK", marks=using('gauxc')),
    ]
) 
@pytest.mark.parametrize("df_basis_scf", [ "CC-PVDZ-JKFIT", "DEF2-UNIVERSAL-JFIT" ]) #to be extended in the future
def test_j_algo_bp86(j_algo, k_algo, df_basis_scf, mols):
    """Test SCF_TYPE={J} and all SCF_TYPE={J}+{K} combinations for a BP86 calculation.
    They should all give the exact same answer (within tolerance).""" 

    molecule = mols["h2o"]
    
    # run base composite J algorithm 
    psi4.set_options({"scf_type" : j_algo, "basis": "cc-pvdz", "df_basis_scf": df_basis_scf})
    energy_dfdirj = psi4.energy("bp86", molecule=molecule) 
    
    # compare composite combinations to base J algorithm
    scf_type = j_algo + "+" + k_algo    
    screening = "CSAM" if any([ _ in scf_type for _ in [ "COSX", "SNLINK" ] ]) else "DENSITY"

    psi4.set_options({"scf_type" : scf_type, "reference": "rhf", "basis": "cc-pvdz", "df_basis_scf": df_basis_scf, "screening": screening})
    energy_composite = psi4.energy("bp86", molecule=molecule) 
 
    assert compare_values(energy_dfdirj, energy_composite, 6, f'BP86/{df_basis_scf} {scf_type} accurate to {j_algo} (1e-6 threshold)')

@pytest.mark.parametrize(
    "scf_type,keywords",
    [
        # should work now
        pytest.param("direct", {"screening" : "density" },id="direct-tuned"),
        pytest.param("mem_df", {"screening" : "schwarz"}, id="memdf-tuned"),
        pytest.param("disk_df", {"screening" : "schwarz"}, id="diskdf-tuned"),
        pytest.param("dfdirj+link", {"screening" : "density", "link_ints_tolerance": 1e-12, }, id="dfdirj+link-tuned"),
        pytest.param("dfdirj+cosx", {"screening" : "schwarz", "cosx_ints_tolerance": 1e-12, "cosx_maxiter_final": 0 }, id="dfdirj+cosx-tuned-1grid"),
        pytest.param("dfdirj+cosx", {"screening" : "schwarz", "cosx_ints_tolerance": 1e-12, "cosx_maxiter_final": -1 }, id="dfdirj+cosx-tuned-2grid"),
        
        # should work by the time we are done
        pytest.param("direct", {}, id="direct-default"),
        pytest.param("mem_df", {}, id="memdf-default"),
        pytest.param("disk_df", {}, id="diskdf-default"),
        pytest.param("dfdirj+link", {}, id="dfdirj+link-default"), # should fail because LINK isnt yet compatible with SCREENING=CSAM
        pytest.param("dfdirj+cosx", {}, id="dfdirj+cosx-default"), # should fail because FISAPT needs full convergence on COSX grid 
    ]
)
@pytest.mark.parametrize(
    "mol",
     [
        pytest.param("eneyne"),
        pytest.param("benzene2"),
    ]
)
def test_fisapt(scf_type, keywords, mol, request):
    test_id = request.node.callspec.id
   
    # TODO: Add reference values 
    ref = {
        "direct": {
            "HF TOTAL ENERGY": 0.0,
            "SAPT TOTAL ENERGY": 0.0,
            "SAPT ELST ENERGY": 0.0,
            "SAPT EXCH ENERGY": 0.0,
            "SAPT IND ENERGY": 0.0,
            "SAPT DISP ENERGY": 0.0,
        },
        "df": {
            "HF TOTAL ENERGY": 0.0,
            "SAPT TOTAL ENERGY": 0.0,
            "SAPT ELST ENERGY": 0.0,
            "SAPT EXCH ENERGY": 0.0,
            "SAPT IND ENERGY": 0.0,
            "SAPT DISP ENERGY": 0.0,
        },
        "dfdirj+link": {
            "HF TOTAL ENERGY": 0.0,
            "SAPT TOTAL ENERGY": 0.0,
            "SAPT ELST ENERGY": 0.0,
            "SAPT EXCH ENERGY": 0.0,
            "SAPT IND ENERGY": 0.0,
            "SAPT DISP ENERGY": 0.0,
        },
        "dfdirj+cosx": {
            "HF TOTAL ENERGY": 0.0,
            "SAPT TOTAL ENERGY": 0.0,
            "SAPT ELST ENERGY": 0.0,
            "SAPT EXCH ENERGY": 0.0,
            "SAPT IND ENERGY": 0.0,
            "SAPT DISP ENERGY": 0.0,
        },
    }
    
    molecule = {
        "eneyne": """
            C   0.000000  -0.667578  -2.124659
            C   0.000000   0.667578  -2.124659
            H   0.923621  -1.232253  -2.126185
            H  -0.923621  -1.232253  -2.126185
            H  -0.923621   1.232253  -2.126185
            H   0.923621   1.232253  -2.126185
            --
            C   0.000000   0.000000   2.900503
            C   0.000000   0.000000   1.693240
            H   0.000000   0.000000   0.627352
            H   0.000000   0.000000   3.963929
        """,
        "benzene2": """
            0 1
            O    -1.3885044    1.9298523   -0.4431206
            H    -0.5238121    1.9646519   -0.0064609
            C    -2.0071056    0.7638459   -0.1083509
            C    -1.4630807   -0.1519120    0.7949930
            C    -2.1475789   -1.3295094    1.0883677
            C    -3.3743208   -1.6031427    0.4895864
            C    -3.9143727   -0.6838545   -0.4091028
            C    -3.2370496    0.4929609   -0.7096126
            H    -0.5106510    0.0566569    1.2642563
            H    -1.7151135   -2.0321452    1.7878417
            H    -3.9024664   -2.5173865    0.7197947
            H    -4.8670730   -0.8822939   -0.8811319
            H    -3.6431662    1.2134345   -1.4057590
            --
            0 1
            O     1.3531168    1.9382724    0.4723133
            H     1.7842846    2.3487495    1.2297110
            C     2.0369747    0.7865043    0.1495491
            C     1.5904026    0.0696860   -0.9574153
            C     2.2417367   -1.1069765   -1.3128110
            C     3.3315674   -1.5665603   -0.5748636
            C     3.7696838   -0.8396901    0.5286439
            C     3.1224836    0.3383498    0.8960491
            H     0.7445512    0.4367983   -1.5218583
            H     1.8921463   -1.6649726   -2.1701843
            H     3.8330227   -2.4811537   -0.8566666
            H     4.6137632   -1.1850101    1.1092635
            H     3.4598854    0.9030376    1.7569489
            symmetry c1
            no_reorient
            no_com
       """,
    }

    psi4.geometry(molecule[mol])

    psi4.set_options({**{
        "scf_type": scf_type,
        "basis": "6-31g", 
        "fisapt_do_fsapt": False,
        "freeze_core": True,
        "ints_tolerance": 1e-12,
        "cphf_r_convergence": 1e-8,
        "d_convergence": 1e-8,
        "e_convergence": 1e-8,
        "maxiter": 30,
        "save_jk": True,
    }, **keywords})

    E, wfn = psi4.energy("fisapt0", return_wfn=True)

    # six things to test: EHF, ESAPT, Electrostatics, Exch, Ind, Disp
    for component, ref_E in ref[scf_type].items():
        pass
        #assert compare_values(ref_E, wfn.variable(component), 6, f'{test_id} accurate to reference') # wfn.variable is TODO for SAPT
        #assert compare_values(ref_E, psi4.variable(component), 6, f'{test_id} accurate to reference')

    # probe underlying JK object - might have to adjust FISAPT code to support
    # correct post-guess method?
    clean_jk_name = wfn.jk().name().replace("-", "") # replace DF-DirJ with DFDirJ
    clean_jk_name = clean_jk_name.replace("DirectJK", "Direct") # DirectJK should be Direct instead
    # TODO add more jk name conditionals (MemDF/DiskDF, CompositeJK) 
    
    assert clean_jk_name == scf_type, f'{test_id} has correct end method'
>>>>>>> Add FISAPT tests for arbitrary JKs in test_compositejk.py
