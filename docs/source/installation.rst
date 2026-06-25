Installation
============

Prepare the Python environment 
-------------------------------------
Either way you choose below, first create and activate a Python 
3.12 environment, for example:

.. code-block:: bash

   conda create -n cellcoloc python=3.12 -y
   conda activate cellcoloc

.. note::

   We have tested CellColoc with Python 3.12. Newer versions may work but 
   are not guaranteed to be compatible. Older Python versions are not supported,
   as CellColoc relies on `OMIO <https://omio.readthedocs.io/en/latest/>`_ for 
   reading microscopy data, and OMIO requires Python 3.12 or newer.


GPU support on Windows
^^^^^^^^^^^^^^^^^^^^^^

On Windows and in case you have a **CUDA-ready GPU**, we recommend installing
PyTorch *prior* to CellColoc in order to prevent the installation of the CPU-only 
PyTorch version for Cellpose. Thus, in case you want to use Cellpose with GPU 
acceleration on your Windows machine, please follow these additional steps *after* 
creating and activating the Python environment described above:

1. If not already done, install or update the `NVIDIA drivers <https://www.nvidia.com/Download/index.aspx>`_.  
   For the standard PyTorch wheels from the PyTorch website, a separate `CUDA Toolkit <https://developer.nvidia.com/cuda-downloads>`_ or `cuDNN <https://developer.nvidia.com/cudnn>`_ installation is often **not** required, because the wheel already bundles the needed CUDA runtime components. If you use a custom CUDA or PyTorch setup, however, those additional installations may still be relevant.
2. Find out your CUDA version by running ``nvidia-smi`` in the terminal.
3. Visit the `PyTorch website <https://pytorch.org/get-started/locally/>`_ and copy 
   the correct command for your system. 
4. In that conda environment you have created earlier, paste the PyTorch installation 
   command you found in step 3. For example, for a system with CUDA 13.0, you would use:

   .. code-block:: bash

      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

5. Verify that PyTorch can access your GPU:

   .. code-block:: bash

      python -c "import torch; print(torch.cuda.is_available())"

Now follow one of the installation methods described below.

  
Install CellColoc from PyPI
---------------------------

The standard installation is:

.. code-block:: bash

   pip install cellcoloc


Interactive use
^^^^^^^^^^^^^^^^

If you want to use CellColoc together with VS Code's interactive window or a
notebook-like workflow, install the interactive extra:

.. code-block:: bash

   pip install "cellcoloc[interactive]"


Cellpose 3
^^^^^^^^^^

CellColoc is designed to work with both older `Cellpose 3 <https://www.cellpose.org>`_ installations and
newer `Cellpose 4 <https://www.cellpose.org>`_ installations.

If you specifically want the tested Cellpose 3 variant, install:

.. code-block:: bash

   pip install "cellcoloc[cellpose3]"

Alternatively, you can install CellColoc first and then pin Cellpose manually:

.. code-block:: bash

   pip install cellcoloc
   pip install "cellpose==3.1.1.2"


Development install
-------------------

For local development from a clone of the repository:

.. code-block:: bash

   git clone https://github.com/FabrizioMusacchio/CellColoc.git
   cd CellColoc
   pip install -e .

For interactive development:

.. code-block:: bash

   pip install -e ".[interactive]"

Updating CellColoc
------------------

To update an existing installation, run:

.. code-block:: bash

   pip install --upgrade cellcoloc

If you are using the interactive extra, run:

.. code-block:: bash

   pip install --upgrade "cellcoloc[interactive]"

If you are using the development install, run, after 
pulling the latest changes from the repository:

.. code-block:: bash

   pip install --upgrade -e .

or, for interactive development:

.. code-block:: bash

   pip install --upgrade -e ".[interactive]"

Removing the CellColoc environment
------------------------------------

In case you want to remove the CellColoc conda environment you have created 
in the steps above, e.g., for a fresh reinstallation, simply execute the following 
command *after* deactivating the environment:

.. code-block:: bash

   conda env remove -n cellcoloc


User-scripts
------------

Please visit CellColoc's `GitHub repository <https://github.com/FabrizioMusacchio/cellcoloc/tree/main>`_ 
for the latest user-scripts. The repository contains a ``user_scripts/`` directory with example scripts 
for 2D and 3D datasets, including single-channel and three-channel analyses. We
discuss them in detail in the `Usage <usage.html>`_ section of the documentation.

Example datasets
----------------

In order to run the example user-scripts, you will need to download the example datasets. 
Please refer to the  `Example data set section <usage_example_datasets.html>`_ 
and follow the download instructions there. 

Dependencies
------------

The core package currently depends on:

- ``cellpose``
- ``omio-microscopy``
- ``matplotlib``
- ``pandas``
- ``openpyxl``
- ``scikit-image``
- ``appdirs``

The optional interactive extra additionally provides:

- ``ipykernel``
