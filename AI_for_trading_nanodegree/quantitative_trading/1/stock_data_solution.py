#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[ ]:


import pandas as pd
import quiz_tests


# ## Quiz Solution

# In[1]:


def csv_to_close(csv_filepath, field_names):
    """Reads in data from a csv file and produces a DataFrame with close data.
    
    Parameters
    ----------
    csv_filepath : str
        The name of the csv file to read
    field_names : list of str
        The field names of the field in the csv file

    Returns
    -------
    close : DataFrame
        Close prices for each ticker and date
    """
    
    # TODO: Implement Function
    
    return pd.read_csv(csv_filepath, names=field_names).pivot(index='date', columns='ticker', values='close')


quiz_tests.test_csv_to_close(csv_to_close)


# In[ ]:




