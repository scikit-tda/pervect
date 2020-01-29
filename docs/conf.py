# -*- coding: utf-8 -*-
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
from pervect import __version__
from sktda_docs_config import *

project = u'PerVect'
copyright = u'2020, Leland McInnes, Colin Weir, Elizabeth Munch'
author = u'Leland McInnes, Colin Weir'

version = __version__
release = __version__

html_theme_options.update({
  # Google Analytics info
  'ga_ua': 'UA-124965309-6',
  'ga_domain': '',
  'gh_url': 'scikit-tda/pervect'
})

html_short_title = project
htmlhelp_basename = 'PerVect'