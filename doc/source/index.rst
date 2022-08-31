.. FEniCSx-Shells documentation master file, created by
   sphinx-quickstart on Fri Aug 26 15:38:02 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to FEniCSx-Shells's documentation!
==========================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. include:: ../../README.md
   :parser: myst_parser.sphinx_

..
  Indices and tables
  ==================

  * :ref:`genindex`
  * :ref:`modindex`
  * :ref:`search`


Documented demos
================

Clamped linear Reissner-Mindlin plate problem using the Duran-Liberman reduction
operator (MITC) to cure shear-locking:

.. toctree::
    :titlesonly:
    :maxdepth: 1

    demo/demo_reissner-mindlin-clamped.md

Clamped linear Reissner-Mindlin plate problem using the Pechstein-Schöberl
TDNNS (tangential displacement normal-normal stress) element to cure
shear-locking:

.. toctree::
    :titlesonly:
    :maxdepth: 1

    demo/demo_reissner-mindlin-clamped-tdnns.md

Simply-supported linear Reissner-Mindlin plate problem using the MITC4
reduction operator to cure shear-locking:

.. toctree::
    :titlesonly:
    :maxdepth: 1

    demo/demo_reissner-mindlin-simply-supported.md
