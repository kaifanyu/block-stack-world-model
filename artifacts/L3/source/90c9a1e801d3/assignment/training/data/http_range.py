"""Bounded HTTP range reader for selective tar/NPZ reads from public artifacts."""
import io
import time
from collections import OrderedDict
import requests


class RangeReader(io.RawIOBase):
    def __init__(self,url,block=262144):
        self.source=url;self.url=url;self.block=block;self.position=0
        self.cache=OrderedDict();self.session=requests.Session();self.downloaded=0
        self.request_count=0;self.request_seconds=0.;self.max_blocks=max(16,16777216//block)
        response=self._get(url,headers={'Range':'bytes=0-0'},timeout=(5,60))
        if response.status_code!=206:raise RuntimeError(f'Range request returned HTTP {response.status_code}')
        self.url=response.url;self.length=int(response.headers['Content-Range'].split('/')[-1])
    def _get(self,url,**kwargs):
        last=None
        for attempt in range(5):
            started=time.monotonic();self.request_count+=1
            try:return self.session.get(url,**kwargs)
            except requests.RequestException as exc:
                last=type(exc).__name__
                if attempt<4:time.sleep(min(2**attempt,10))
            finally:self.request_seconds+=time.monotonic()-started
        raise RuntimeError(f'HTTP request failed after5attempts ({last}); URL omitted')
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        self.position=offset if whence==0 else self.position+offset if whence==1 else self.length+offset
        if self.position<0:raise ValueError('Negative offset')
        return self.position
    def read(self,size=-1):
        end=self.length if size<0 else min(self.position+size,self.length)
        parts=[]
        while self.position<end:
            start=self.position//self.block*self.block
            if start not in self.cache:
                stop=min(start+self.block,self.length)-1
                for attempt in range(5):
                    response=self._get(self.url,headers={'Range':f'bytes={start}-{stop}'},timeout=(5,90))
                    if response.status_code==206:break
                    if response.status_code==403:
                        refresh=self._get(self.source,headers={'Range':'bytes=0-0'},timeout=(5,60));self.url=refresh.url
                    time.sleep(min(2**attempt,10))
                if response.status_code!=206 or response.headers.get('Content-Range','').split('/')[0]!=f'bytes {start}-{stop}':
                    raise RuntimeError(f'Invalid range response HTTP {response.status_code} at {start}')
                chunk=response.content
                if len(chunk)!=stop-start+1:raise IOError('Short HTTP range')
                self.downloaded+=len(chunk);self.cache[start]=chunk
                while len(self.cache)>self.max_blocks:self.cache.popitem(last=False)
            self.cache.move_to_end(start)
            chunk=self.cache[start];offset=self.position-start
            take=min(len(chunk)-offset,end-self.position)
            parts.append(chunk[offset:offset+take]);self.position+=take
        return b''.join(parts)
    def prefetch(self,start,size):
        """Fetch a bounded contiguous ZIP field region in one HTTP request."""
        if size<=0:return
        lo=start//self.block*self.block
        hi=min(self.length,((start+size+self.block-1)//self.block)*self.block)
        if all(k in self.cache for k in range(lo,hi,self.block)):return
        stop=hi-1
        for attempt in range(5):
            response=self._get(self.url,headers={'Range':f'bytes={lo}-{stop}'},timeout=(5,90))
            if response.status_code==206:break
            if response.status_code==403:
                refresh=self._get(self.source,headers={'Range':'bytes=0-0'},timeout=(5,60));self.url=refresh.url
            time.sleep(min(2**attempt,10))
        if response.status_code!=206 or response.headers.get('Content-Range','').split('/')[0]!=f'bytes {lo}-{stop}':
            raise RuntimeError(f'Invalid prefetch response HTTP {response.status_code} at {lo}')
        data=response.content
        if len(data)!=hi-lo:raise IOError('Short HTTP prefetch')
        self.downloaded+=len(data)
        for offset in range(lo,hi,self.block):
            self.cache[offset]=data[offset-lo:offset-lo+self.block]
            self.cache.move_to_end(offset)
        while len(self.cache)>self.max_blocks:self.cache.popitem(last=False)
    def close(self):
        self.session.close();super().close()
