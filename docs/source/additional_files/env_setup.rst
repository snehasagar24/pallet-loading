.. _conda-environment-ref:

Create Work Environment
=======================

1. Install Anaconda: Install_Anaconda_.

.. _Install_Anaconda: https://docs.anaconda.com/anaconda/install/

2. Managing Conda Environments: Manage_Environments_.

.. _Manage_Environments: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

3. Example of creating an environment with Python 3.6 (used for this solution)

    .. code-block:: python

       conda create -n palletLoading python=3.6

4. Activating and deactivating environment for Windows

    #. Conda >= 4.6

	    .. code-block:: python

		   conda activate palletLoading

		   conda deactivate palletLoading

    #. Conda < 4.6

	    .. code-block:: python

		   activate palletLoading

		   deactivate palletLoading

4. Activating and deactivating environment for Linux and MacOS

	.. code-block:: python

		source activate palletLoading

		source deactivate palletLoading
	  
	   

