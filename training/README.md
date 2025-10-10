Structure-Preserving Models with Calibrated Uncertainty
=======================================================

Implementation of "Comparing UDEs and BNODEs for Microgrid Dynamics"

Algorithms
----------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   FileAlgorithmMethodOutputalgo1_ude_training.jlUDE Training (Alg 1)L-BFGS grid searchcheckpoints/ude/algo2_bnode_inference.jlBNODE Inference (Alg 2)MCMC/NUTSbnode_posterior.bsonalgo3_bnode_calibration.jlBNODE Calibration (Alg 3)Variance scalingbnode_calibrated.jsonalgo4_ude_calibration.jlUDE Calibration (Alg 4)Conformal predictionude_calibration.json   `

Usage
-----

bash

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`julia algo1_ude_training.jl        julia algo2_bnode_inference.jl     julia algo3_bnode_calibration.jl   julia algo4_ude_calibration.jl`   

Data Structure
--------------

Place in data/:

*   training\_roadmap.csv (50 scenarios)
    
*   validation\_roadmap.csv (10 scenarios)
    
*   test\_roadmap.csv (10 scenarios)
    

**Columns:** scenario, t, x1, x2, u, d, Pgen, Pload, ηin, ηout, α, β, γ

Key Results (Table 3)
---------------------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   ModelRMSE x₂Coverage@90%InferenceUDE0.247586.8%0.27 msBNODE (calibrated)-89.3%-Physics0.2520-0.08 ms   `

Dependencies
------------

julia

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   DifferentialEquations, Turing, Optim, CSV, DataFrames, BSON, JSON3   `
