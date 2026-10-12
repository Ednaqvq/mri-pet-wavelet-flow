"""Download and SHA-256 verify one source/seed from GitHub Releases."""
import argparse,json,hashlib,tarfile,urllib.request,shutil
from pathlib import Path
URL='https://github.com/Ednaqvq/mri-pet-wavelet-flow/releases/download/v1.0.0/'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',choices=['ADNI','NACC'],required=True);p.add_argument('--seed',type=int,choices=[42,2026,3407,1234,5678],default=42);p.add_argument('--output',default='weights');a=p.parse_args()
 manifest=json.load(urllib.request.urlopen(URL+'weights_manifest.json'))
 item=next(x for x in manifest['weights'] if x['cohort']==a.source and x['seed']==a.seed)
 out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True);dest=out/f'{a.source}_seed{a.seed}'
 if dest.exists():raise FileExistsError(dest)
 archive=out/item['asset'];tmp=archive.with_suffix(archive.suffix+'.part')
 with urllib.request.urlopen(URL+item['asset']) as src,open(tmp,'wb') as f:shutil.copyfileobj(src,f)
 if sha(tmp)!=item['sha256']:raise ValueError('Archive SHA-256 mismatch')
 tmp.replace(archive)
 with tarfile.open(archive) as tar:
  for member in tar.getmembers():
   if not (out/member.name).resolve().is_relative_to(dest) or not (member.isdir() or member.isfile()):raise ValueError('Unsafe archive member')
  for member in tar.getmembers():
   path=out/member.name
   if member.isdir():path.mkdir(parents=True,exist_ok=True)
   else:
    path.parent.mkdir(parents=True,exist_ok=True)
    with tar.extractfile(member) as src,open(path,'wb') as f:shutil.copyfileobj(src,f)
 for file,info in item['files'].items():
  if sha(dest/file)!=info['sha256']:raise ValueError(f'Weight SHA-256 mismatch: {file}')
 print(dest)
if __name__=='__main__':main()
