// GitHub Pages 배포: 정적 API 스냅샷 생성 → Pages용 빌드 → gh-pages 브랜치에 push
//   cd frontend-web && npm run deploy:pages
//   (Python 경로 지정: set PYTHON=C:\Python39\python.exe)
import { execSync } from 'node:child_process';
import { cpSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const webDir = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const repoDir = resolve(webDir, '..');
const run = (cmd, cwd, env = {}) =>
  execSync(cmd, { cwd, stdio: 'inherit', env: { ...process.env, ...env } });
const read = (cmd, cwd) => execSync(cmd, { cwd }).toString().trim();

const python = process.env.PYTHON || 'python';
const remote = read('git remote get-url origin', repoDir);
const repoName = remote.replace(/\.git$/, '').split('/').pop();
const commit = read('git rev-parse --short HEAD', repoDir);

console.log('\n[1/3] Exporting static API data...');
run(`"${python}" -X utf8 scripts/export_static_api.py`, join(repoDir, 'backend'));

console.log(`\n[2/3] Building for /${repoName}/ ...`);
run('npm run build', webDir, { VITE_BASE_PATH: `/${repoName}/`, VITE_STATIC_HOST: '1' });

console.log('\n[3/3] Publishing to gh-pages branch...');
const tmp = mkdtempSync(join(tmpdir(), 'mlstudio-pages-'));
try {
  cpSync(join(webDir, 'dist'), tmp, { recursive: true });
  writeFileSync(join(tmp, '.nojekyll'), '');
  run('git init -q -b gh-pages', tmp);
  run('git add -A', tmp);
  run(`git -c core.autocrlf=false commit -q -m "Deploy ${commit}"`, tmp);
  run(`git push -f "${remote}" gh-pages`, tmp);
} finally {
  rmSync(tmp, { recursive: true, force: true });
}
console.log(`\nDone → https://${remote.split('/').at(-2)}.github.io/${repoName}/`);
