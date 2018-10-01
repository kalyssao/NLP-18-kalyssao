
# coding: utf-8

# In[ ]:


def m_e_d(source, target):
    if(source == ""):
        return len(target)
        
    if(target == ""):
        return len(source)
        
    if(source[-1] == target[-1]):
        cost = 0   
    else:
        cost = 2
            
    med = min([m_e_d(source[:-1], target) + 1,
               m_e_d(source, target[:-1]) + 1, 
               m_e_d(source[:-1], target[:-1]) + cost])
    
    return med


# In[ ]:


m_e_d('','ii')


# In[ ]:


print("Minimum edit distance between intention and execution is" m_e_d("intention","execution"))

