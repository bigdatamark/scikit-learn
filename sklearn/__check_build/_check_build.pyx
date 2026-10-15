# [MODULE DOC]
# Minimal compiled extension whose only purpose is to prove that scikit-learn's
# compiled (Cython) modules were built and can be imported. The package
# __init__ imports it and turns an ImportError into a helpful build message.
def check_build():
    # [FUNCTION DOC]
    # Behavior: does nothing; successful import/call shows the extension loads.
    # Inputs: none. Outputs: None. Exceptions: none. Side effects: none.
    return
