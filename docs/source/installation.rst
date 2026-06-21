Installation
============

PyPI
----

The standard installation is:

.. code-block:: bash

   pip install cellcoloc


Interactive use
---------------

If you want to use CellColoc together with VS Code's interactive window or a
notebook-like workflow, install the interactive extra:

.. code-block:: bash

   pip install "cellcoloc[interactive]"


Cellpose 3
----------

CellColoc is designed to work with both older Cellpose 3 installations and
newer Cellpose 4 installations.

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


Python version
--------------

CellColoc is currently prepared for Python 3.12 and newer.


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

