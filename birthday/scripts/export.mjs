import { cp, readdir, readFile, writeFile, rm } from 'node:fs/promises';
import path from 'node:path';

// Keep a directly accessible static page alongside its editable source.
// GitHub Pages' Jekyll skips _next folders, so give the bundle a normal name.
const root = process.cwd();
async function publish(from, to) {
  await cp(from, to, { recursive: true });
  async function rewrite(dir) {
    for (const entry of await readdir(dir, { withFileTypes: true })) {
      const file = path.join(dir, entry.name);
      if (entry.isDirectory()) await rewrite(file);
      else if (/\.(html|js|css|json|txt)$/.test(entry.name)) {
        const source = await readFile(file, 'utf8');
        await writeFile(file, source.replaceAll('_next/', 'assets/'));
      }
    }
  }
  await rewrite(to);
}
// Replace generated bundles so removed content is not kept in stale assets.
await rm(path.join(root, 'assets'), { recursive: true, force: true });
await publish(path.join(root, 'out/_next'), path.join(root, 'assets'));
for (const name of ['index.html', '404.html']) {
  const source = await readFile(path.join(root, 'out', name), 'utf8');
  await writeFile(path.join(root, name), source.replaceAll('_next/', 'assets/'));
}
console.log('Ready: birthday/index.html and birthday/assets/');
