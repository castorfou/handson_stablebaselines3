#!/usr/bin/env python
# coding: utf-8

# In[1]:


def getFactors(n):
    # Create an empty list for factors
    factors=[];

    # Loop over all factors
    for i in range(1, n + 1):
        if n % i == 0:
            factors.append(i)

    # Return the list of factors
    return factors

# Call the function with a given value
print (getFactors(256))


# In[3]:


sum(getFactors(256))/2


# In[8]:


for i in range(100000):
    diviseurs = getFactors(i)
    if (i == sum(diviseurs)/2) :
        print(f'{i} est parfait, et les diviseurs sont {diviseurs}')
    if (i % 100 ==0):
        print(i)


# In[9]:


def divisorGen(n):
    factors = list(factorGenerator(n))
    nfactors = len(factors)
    f = [0] * nfactors
    while True:
        yield reduce(lambda x, y: x*y, [factors[x][0]**f[x] for x in range(nfactors)], 1)
        i = 0
        while True:
            f[i] += 1
            if f[i] <= factors[i][1]:
                break
            f[i] = 0
            i += 1
            if i >= nfactors:
                return


# In[ ]:


for i in range(100000):
    diviseurs = getFactors(i)
    if (i == sum(diviseurs)/2) :
        print(f'{i} est parfait, et les diviseurs sont {diviseurs}')
    if (i % 100 ==0):
        print(i)


# In[11]:


from sympy import divisors

divisors(496)


# In[15]:


for i in range(10000000):
    diviseurs = divisors(i)
    if (i == sum(diviseurs)/2) :
        print(f'{i} est parfait, et les diviseurs sont {diviseurs}')
    if (i % 100000 ==0):
        print(i)


# In[16]:


divisors(6)


# In[17]:


divisors(28)


# In[18]:


divisors(496)


# In[19]:


divisors(8128)


# In[20]:


divisors(33550336)


# In[22]:


divisors(8589869056)


# In[23]:


divisors(137438691328)


# In[24]:


divisors(2305843008139952128)


# In[25]:


divisors(2658455991569831744654692615953842176)


# In[26]:


divisors(2**16*(2**17-1))


# In[27]:


2**16*(2**17-1)


# In[28]:


2**57885160*(2**57885161 - 1)


# In[ ]:




