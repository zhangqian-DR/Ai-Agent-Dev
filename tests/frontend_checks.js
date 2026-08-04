/**
 * 页面逻辑的检查。由 tests/test_frontend.py 驱动，也可以自己跑：
 *     node tests/frontend_checks.js [index.html 的路径]
 *
 * 为什么是这个形状：项目是纯 Python 的，双击 .bat 就能用，不该为了测前端引入
 * 一整套 JS 工具链。所以这里不装任何依赖——把 index.html 里的 <script> 抠出来，
 * 配一套最小的 DOM 假件直接 eval，然后驱动那几个函数。
 *
 * 每条检查输出一行 `RESULT<TAB>PASS|FAIL<TAB>名字<TAB>细节`，pytest 那边按行
 * 解析成一个个用例，红了能直接看出是哪条。
 */
const fs = require('fs');
const path = require('path');

// ---------- 最小 DOM 假件 ----------

function node(tag) {
  return {
    _tag: tag, children: [], className: '', textContent: '', style: {},
    dataset: {}, title: '', disabled: false, scrollTop: 0, scrollHeight: 0,
    _on: {},
    appendChild(c) { this.children.push(c); return c; },
    replaceChild(n, o) { const i = this.children.indexOf(o); if (i >= 0) this.children[i] = n; },
    get lastChild() { return this.children[this.children.length - 1]; },
    set innerHTML(v) { if (v === '') this.children = []; },
    get innerHTML() { return ''; },
    addEventListener(ev, fn) { (this._on[ev] = this._on[ev] || []).push(fn); },
    click() { (this._on.click || []).forEach(f => f()); },
  };
}

let ids = {};
globalThis.document = {
  createElement: node,
  getElementById(id) { return (ids[id] = ids[id] || node('div')); },
};

let HISTORY = { messages: [] };
let MEMORIES = { memories: [] };
let POLL = { events: [], pending: null, running: false, plan: [], plan_current: 0,
             work_dir: 'D:/ws', path: '', model: 'qwen-plus', max_steps: 20 };
const DELETED = [];
globalThis.fetch = (url, opts) => {
  const u = String(url);
  if (opts && opts.method === 'DELETE') {
    DELETED.push(u);
    MEMORIES = { memories: MEMORIES.memories.filter(m => !u.endsWith('/' + m.id)) };
    return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve({ ok: true }) });
  }
  const body = u.includes('/memories') ? MEMORIES : u.includes('/poll') ? POLL : HISTORY;
  return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(body) });
};
globalThis.setTimeout = globalThis.setTimeout;   // 轮询会排下一次，进程退出时一起收

// ---------- 载入页面脚本 ----------

const HTML = process.argv[2] ||
  path.join(__dirname, '..', 'app', 'web', 'static', 'index.html');
const src = fs.readFileSync(HTML, 'utf8').match(/<script>([\s\S]*?)<\/script>/)[1];

let seq = 0;
function load() {
  ids = {};                                  // 每个场景一套干净的 DOM
  const name = '__api' + (++seq);
  // 把 IIFE 收尾那句换成导出，顺便不让它启动 1 秒轮询
  eval(src.replace('setCtx(0);poll();',
    `globalThis.${name}={handle:handle,restore:restore,showGate:showGate,poll:poll};`));
  return { api: globalThis[name], log: ids['log'], planEl: ids['plan'],
           mem: ids['mem'], pathEl: ids['path'] };
}

const text = n => (n.textContent || '') + n.children.map(text).join(' ');

let failed = 0;
function check(name, cond, detail) {
  const d = cond ? '' : JSON.stringify(detail);
  console.log(`RESULT\t${cond ? 'PASS' : 'FAIL'}\t${name}\t${d}`);
  if (!cond) failed++;
}

// ---------- 计划面板 ----------
{
  const { api, log, planEl } = load();
  const cards = () => log.children.filter(c => text(c).includes('update_plan')).length;
  const marks = () => planEl.children.map(li => li.className);

  api.handle({ type: 'plan', steps: ['查资料', '写代码'], current: 1 });
  const afterFirst = cards();
  api.handle({ type: 'plan', steps: ['查资料', '写代码'], current: 2 });
  check('推进一步不重画计划卡', afterFirst === 1 && cards() === 1, cards());
  check('推进后高亮第 2 步', marks()[0] === 'done' && marks()[1] === 'active', marks());

  // 整份换掉计划时必须留痕：面板会悄悄变成新的，聊天流里得有对应的卡
  api.handle({ type: 'plan', steps: ['改方案', '重写', '自测'], current: 1 });
  check('换一份计划会画新卡', cards() === 2, cards());

  api.handle({ type: 'final', content: '搞定', ok: true });
  check('实时 final 划完所有步骤', marks().every(c => c === 'done'), marks());
}

// ---------- 事件类型看标志，不看开头几个字 ----------
{
  const { api, log } = load();
  api.handle({ type: 'final', content: '出错：这三个字开头的正常回答', ok: true });
  const b = log.children[0];
  check('正常回答以「出错：」开头也不判成错误',
        b.className.includes('msg-final') && b.children[0].textContent === '✅', b.className);
}
{
  const { api, log } = load();
  api.handle({ type: 'final', content: '出错：RuntimeError', ok: false });
  check('ok=false 才是错误', log.children[0].className.includes('msg-err'),
        log.children[0].className);
}
{
  const { api, log } = load();
  api.handle({ type: 'tool', name: 'read_file', result: '用户拒绝了吗？这是文件内容', ok: true });
  check('工具结果以「用户拒绝」开头也不判成已拒绝',
        !text(log.children[0]).includes('已拒绝'), text(log.children[0]).slice(0, 40));
}
{
  const { api, log } = load();
  api.handle({ type: 'tool', name: 'write_file', result: '用户拒绝了该操作，已跳过。', ok: false });
  check('ok=false 的工具标成已拒绝', text(log.children[0]).includes('已拒绝'),
        text(log.children[0]).slice(0, 40));
}
{
  const { api, log } = load();
  api.handle({ type: 'tool', name: '验收 pytest -q', result: '[exit=1]\nFAILED',
               ok: false, badge: '未通过' });
  const t = text(log.children[0]);
  check('验收失败说「未通过」不说「已拒绝」',
        t.includes('未通过') && !t.includes('已拒绝'), t.slice(0, 60));
}

// ---------- 确认卡片：整盘门控 ----------
{
  const { api, log } = load();
  api.showGate({ id: 1, actions: [
    { name: 'write_file', preview: '--- a.txt\n+++ a.txt\n@@ -0,0 +1 @@\n+A' },
    { name: 'write_file', preview: '--- b.txt\n+++ b.txt\n@@ -0,0 +1 @@\n+B' },
  ]});
  const t = text(log.children[0]);
  check('一批操作画在同一张卡上', t.includes('+A') && t.includes('+B'), t.slice(0, 80));
  check('卡上说明有几个操作', t.includes('2 个操作待确认'), t.slice(0, 60));
  check('按钮是「全部确认/全部拒绝」',
        t.includes('全部确认') && t.includes('全部拒绝'), t.slice(0, 120));
  check('只有一对按钮', (t.match(/全部确认/g) || []).length === 1, t.slice(0, 120));
}
{
  const { api, log } = load();
  api.showGate({ id: 1, actions: [{ name: 'run_command', preview: '$ npm test' }]});
  const t = text(log.children[0]);
  check('单个操作仍是原来的措辞', t.includes('执行命令') && t.includes('确认执行'), t.slice(0, 80));
  check('单个操作不显示序号', !t.includes('1. run_command'), t.slice(0, 80));
}
{
  // 相邻两轮各有一个危险操作时，pending 变空的窗口比 1 秒轮询短——
  // 靠 id 判重，靠「当前有没有卡」的话第二张永远画不出来
  const { api, log } = load();
  const p = { id: 7, actions: [{ name: 'write_file', preview: '+x' }] };
  api.showGate(p);
  api.showGate(p);
  check('同 id 不重复画卡', log.children.length === 1, log.children.length);
  api.showGate({ id: 8, actions: [{ name: 'write_file', preview: '+y' }] });
  check('新 id 要画新卡', log.children.length === 2, log.children.length);
}

// ---------- 回放：新记录看 status，老记录退回去看开头 ----------
const replay = (messages, then) => {
  const ctx = load();
  HISTORY = { title: 't', messages };
  return ctx.api.restore().then(() => then(ctx));
};

replay([
  { role: 'plan', content: JSON.stringify({ steps: ['一', '二', '三'], current: 2 }), tool: '', status: '' },
  { role: 'final', content: '出错：RuntimeError: 炸了', tool: '', status: 'error' },
], ({ log, planEl }) => {
  const marks = planEl.children.map(li => li.className);
  check('回放 final 划完所有步骤', marks.length === 3 && marks.every(c => c === 'done'), marks);
  const err = log.children.find(c => (c.className || '').includes('msg-err'));
  check('回放 status=error 用 ✖', err && err.children[0].textContent === '✖',
        err ? err.children[0].textContent : '没找到错误气泡');
}).then(() => replay([
  { role: 'final', content: '出错：老记录，没有 status', tool: '', status: '' },
], ({ log }) => {
  check('老记录（status 为空）仍靠开头兜底判成错误',
        log.children[0].className.includes('msg-err'), log.children[0].className);
})).then(() => replay([
  { role: 'tool', content: '用户拒绝了该操作，已跳过。', tool: 'write_file', status: 'rejected' },
], ({ log }) => {
  check('回放 status=rejected 标成已拒绝', text(log.children[0]).includes('已拒绝'),
        text(log.children[0]).slice(0, 40));

// ---------- 记忆面板 ----------

// ---------- 状态栏：分诊结果要看得见 ----------
})).then(() => {
  const { api, pathEl } = load();
  POLL = { ...POLL, path: 'slow', model: 'qwen-max' };
  api.poll();
  return new Promise(r => setTimeout(r, 0)).then(() => {
    check('状态栏显示分诊到的路径', pathEl.textContent.includes('slow'), pathEl.textContent);
  });
}).then(() => {
  const { api, pathEl } = load();
  POLL = { ...POLL, path: '' };            // 还没发过任务
  api.poll();
  return new Promise(r => setTimeout(r, 0)).then(() => {
    check('还没分诊时显示占位而不是 undefined',
          pathEl.textContent === '—', pathEl.textContent);
  });

// ---------- 记忆面板 ----------
}).then(() => {
  MEMORIES = { memories: [
    { id: 7, fact: '用户主要写 Java', is_negative: false },
    { id: 8, fact: '用 tab 缩进', is_negative: true },
  ]};
  const { mem } = load();
  return new Promise(r => setTimeout(r, 0)).then(() => {
    const items = mem.children.map(li => ({ cls: li.className, txt: text(li) }));
    check('记忆面板渲染出两条', items.length === 2, items);
    check('禁令带 neg 样式和前缀',
          items[1].cls === 'neg' && items[1].txt.includes('禁止'), items[1]);
    check('普通事实不带 neg', !items[0].cls && !items[0].txt.includes('禁止'), items[0]);
    check('不渲染成 [object Object]', items[0].txt.includes('Java'), items[0].txt);

    // 删除是两步的：硬删没有撤销，而这个按钮就在侧栏里，误点代价太大
    const x = mem.children[0].children[1];
    check('每条记忆带一个删除按钮', !!x && x.className.includes('mem-x'), x && x.className);
    DELETED.length = 0;
    x.click();
    check('第一次点只是待确认，不发请求',
          DELETED.length === 0 && x.textContent.includes('删除'),
          { n: DELETED.length, t: x.textContent });
    x.click();
    return new Promise(r => setTimeout(r, 0)).then(() => {
      check('第二次点才真的删',
            DELETED.length === 1 && DELETED[0].endsWith('/7'), DELETED);
      check('删完面板会刷新', mem.children.length === 1, mem.children.length);
    });
  });
}).then(() => {
  console.log(`RESULT\tDONE\t${failed}`);
  process.exit(failed ? 1 : 0);
});
