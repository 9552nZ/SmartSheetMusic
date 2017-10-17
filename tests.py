import time

t = time.time()
featuretime  = sf.hop_size/sf.samplerate
ospeed = 43
dist = sf.debug_dist[200]
[ntrue, nlive] = dist.shape
for j in range(0,40):
    speed = ospeed+j-20
    offset = featuretime*speed
    p1 = np.exp(-dist)
    for i in range(nlive-1):
        n = int((nlive-i)*offset)
        if n>0:
            p1[n:,[i]] = p1[0:-n,[i]]
            p1[0:n,[i]] = 0.1            
        p = np.prod(p1,axis=1)
        p /= np.sum(p)
    if j==0:
        probs = p
    else:
        probs = np.concatenate([probs,p])
        
#probs = probs.reshape([ntrue, -1])
order = probs.argsort()
val = probs[order[-3:]]
spidx = np.sum(val*np.floor(order[-3:]/ntrue))/np.sum(val)
deltaidx = np.sum(val*np.mod(order[-3:], ntrue))/np.sum(val)

print('Speed:',ospeed+spidx-20, 'Delta:', deltaidx, 'Time:',time.time() - t)
