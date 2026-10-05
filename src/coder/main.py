#!/usr/bin/env python
import sys
import warnings
import os
from datetime import datetime

from coder.crew import Coder

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

DEFAULT_ASSIGNMENT = 'Write a python program to calculate the first 10,000 terms \
    of this series, multiplying the total by 4: 1 - 1/3 + 1/5 - 1/7 + ...'

def run():
    """
    Run the crew.
    """
    assignment = ' '.join(sys.argv[1:]) or os.getenv('CODING_ASSIGNMENT') or DEFAULT_ASSIGNMENT
    inputs = {'assignment': assignment}
    
    result = Coder().crew().kickoff(inputs=inputs)
    print(result.raw)




