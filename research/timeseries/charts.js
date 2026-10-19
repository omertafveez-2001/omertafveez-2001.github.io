/* Interactive schematics and figure-derived charts. No experiment is rerun. */
(() => {
    'use strict';
    const data = window.TIMESERIES_CHARTS;
    if (!data) return;
    const NS = 'http://www.w3.org/2000/svg';
    const create = (tag, attributes = {}, text, svg = false) => {
        const node = svg ? document.createElementNS(NS, tag) : document.createElement(tag);
        Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
        if (text != null) node.textContent = text;
        return node;
    };
    const e = (tag, attrs, text) => create(tag, attrs, text);
    const s = (tag, attrs, text) => create(tag, attrs, text, true);
    const text = (svg, x, y, label, attrs = {}) => svg.append(s('text', { x, y, class: 'ts-axis', ...attrs }, label));
    const number = value => Math.abs(value) > 0 && Math.abs(value) < .001 ? value.toFixed(6).replace(/0+$/, '') : Number(value.toFixed(4)).toString();
    function mount(id, root) {
        const figure = document.getElementById(id), original = figure?.querySelector('.figure-image');
        if (!original) return;
        const details = e('details', { class: 'chart-original' });
        details.append(e('summary', {}, 'View original graphic'));
        figure.insertBefore(root, original);
        original.replaceWith(details);
        details.append(original);
    }
    function shell(title, note) {
        const root = e('div', { class: 'interactive-figure ts-interactive' });
        root.append(e('p', { class: 'ts-chart-title' }, title));
        const toolbar = e('div', { class: 'chart-toolbar' });
        const plot = e('div', { class: 'ts-plot' });
        const status = e('p', { class: 'ts-readout', role: 'status' }, 'Hover, tap, or focus a mark to explore.');
        root.append(toolbar, plot, status, e('p', { class: 'chart-note' }, note));
        return { root, toolbar, plot, status };
    }
    function mark(node, description, status) {
        node.setAttribute('tabindex', '0');
        node.setAttribute('role', 'img');
        node.setAttribute('aria-label', description);
        node.classList.add('ts-mark');
        node.append(s('title', {}, description));
        const show = () => { status.textContent = description; };
        node.addEventListener('pointerenter', show);
        node.addEventListener('focus', show);
        node.addEventListener('click', show);
    }
    function csvDownload(filename, rows) {
        const content = rows.map(row => row.map(v => `"${String(v).replaceAll('"', '""')}"`).join(',')).join('\r\n');
        const link = e('a', { href: URL.createObjectURL(new Blob([content], { type: 'text/csv;charset=utf-8' })), download: filename });
        link.click();
        setTimeout(() => URL.revokeObjectURL(link.href), 1000);
    }
    function button(label, action, className = 'chart-download') {
        const node = e('button', { type: 'button', class: className }, label);
        node.addEventListener('click', action);
        return node;
    }
    function zoomControls(toolbar, redraw) {
        const zoom = e('input', { type: 'range', min: '1', max: '4', step: '.25', value: '1', 'aria-label': 'Zoom' });
        const position = e('input', { type: 'range', min: '0', max: '100', step: '1', value: '0', 'aria-label': 'Horizontal position' });
        const zoomLabel = e('label', { class: 'ts-slider' }, 'Zoom');
        const posLabel = e('label', { class: 'ts-slider' }, 'Position');
        zoomLabel.append(zoom);
        posLabel.append(position);
        const output = e('output', { class: 'ts-zoom-value' }, '1×');
        const update = () => { output.textContent = `${Number(zoom.value)}×`; position.disabled = Number(zoom.value) === 1; redraw(); };
        zoom.addEventListener('input', update);
        position.addEventListener('input', redraw);
        position.disabled = true;
        toolbar.append(zoomLabel, output, posLabel, button('Reset', () => { zoom.value = '1'; position.value = '0'; update(); }));
        const read = domain => {
            const span = (domain[1]-domain[0])/Number(zoom.value);
            const start = domain[0] + (domain[1]-domain[0]-span)*Number(position.value)/100;
            return [start, start+span];
        };
        read.state = () => ({ zoom: Number(zoom.value), position: Number(position.value) });
        read.set = (scale, offset) => {
            zoom.value = String(Math.max(1, Math.min(Number(zoom.max || zoom.getAttribute('max')), scale)));
            position.value = String(Math.max(0, Math.min(100, offset)));
            update();
        };
        read.zoomInput = zoom;
        read.positionInput = position;
        return read;
    }
    function dataTable(root, headers, rows) {
        const details = e('details', { class: 'chart-data' });
        details.append(e('summary', {}, 'View figure readings'));
        const wrap = e('div', { class: 'chart-table-wrap', tabindex: '0', role: 'region', 'aria-label': 'Figure readings table' });
        const table = e('table'), head = e('thead'), tr = e('tr'), body = e('tbody');
        headers.forEach(label => tr.append(e('th', { scope: 'col' }, label)));
        head.append(tr);
        rows.forEach(row => { const tr = e('tr'); row.forEach((cell, i) => tr.append(e(i ? 'td' : 'th', i ? {} : { scope: 'row' }, cell))); body.append(tr); });
        table.append(head, body);
        wrap.append(table);
        details.append(wrap);
        root.append(details);
    }
    function heatmap() {
        const ui = shell('Break locations across forecast horizons',
            'Break cells reconstructed from the saved heatmap, using its 604-entry matrix. “Observed” means algorithm-detected boundaries in the observations. One break per horizon in each series; positions are figure readings.');
        const select = e('select', { 'aria-label': 'Forecast horizon' });
        select.append(e('option', { value: 'all' }, 'All 24 horizons'));
        for (let i = 0; i < 24; i++) select.append(e('option', { value: String(i) }, `Horizon ${i}`));
        select.addEventListener('change', draw);
        ui.toolbar.append(select);
        const domain = zoomControls(ui.toolbar, draw);
        function draw() {
            const [min, max] = domain([0, 604]);
            const svg = s('svg', { viewBox: '0 0 760 490', class: 'ts-svg', role: 'group', 'aria-label': 'Predicted and observed break positions; horizontal axis is downsampled entry order.' });
            const x = value => 65+(value-min)/(max-min)*650;
            const horizons = select.value === 'all' ? Array.from({ length: 24 }, (_, i) => i) : [Number(select.value)];
            [['Predicted', data.predictedBreaks, '#a50026', '#fff5f0'], ['Observed', data.observedBreaks, '#08306b', '#f7fbff']].forEach(([name, points, color, background], pi) => {
                const top = 36+pi*226, rowHeight = 166/horizons.length;
                text(svg, 65, top-12, name, { class: 'ts-svg-heading' });
                svg.append(s('rect', { x: 65, y: top, width: 650, height: 166, fill: background, stroke: '#c9c3ba' }));
                horizons.forEach((h, row) => {
                    const cy = top+(row+.5)*rowHeight;
                    if (h % 5 === 0 || horizons.length === 1 || h === 23) text(svg, 55, cy+4, String(h), { 'text-anchor': 'end' });
                    if (points[h] < min || points[h] > max) return;
                    const group = s('g');
                    group.append(s('rect', { x: x(points[h])-.6, y: top+row*rowHeight, width: Math.max(1.2, 650/(max-min)), height: rowHeight, fill: color }));
                    group.append(s('rect', { x: x(points[h])-7, y: top+row*rowHeight, width: 14, height: rowHeight, fill: 'transparent', class: 'ts-hit' }));
                    mark(group, `Horizon ${h} · ${name} break: entry ${points[h]} (read from figure). Predicted ${data.predictedBreaks[h]}; observed ${data.observedBreaks[h]}.`, ui.status);
                    svg.append(group);
                });
                const ticks = Number(max-min) > 400 ? [0,100,200,300,400,500,600] : Array.from({ length: 5 }, (_, i) => min+(max-min)*i/4);
                ticks.filter(v => v >= min && v <= max).forEach(v => text(svg, x(v), top+185, String(Math.round(v)), { 'text-anchor': 'middle' }));
            });
            text(svg, 390, 480, 'Downsampled forecast-entry index', { 'text-anchor': 'middle' });
            text(svg, 18, 245, 'Forecast horizon', { transform: 'rotate(-90 18 245)', 'text-anchor': 'middle' });
            ui.plot.replaceChildren(svg);
        }
        const rows = data.predictedBreaks.map((p, i) => [i, p, data.observedBreaks[i], 'Reconstructed from figure']);
        ui.toolbar.append(button('Export figure readings ↓', () => csvDownload('timeseries-break-readings.csv', [['Horizon','Predicted entry','Observed entry','Provenance'], ...rows])));
        dataTable(ui.root, ['Horizon', 'Predicted entry', 'Observed entry', 'Provenance'], rows);
        draw();
        mount('ts-break-heatmap', ui.root);
    }
    function histogram(id, chart) {
        const ui = shell(chart.title, chart.note);
        if (chart.approximateX) ui.root.insertBefore(e('p', { class: 'ts-reading-note' },
            id === 'trust-score' ? 'Approximate figure readings · original numeric results were not saved.' : 'Slope-bin boundaries are approximate readings of the saved figure.'), ui.toolbar);
        const domain = zoomControls(ui.toolbar, draw);
        const legend = e('div', { class: 'chart-legend' }), active = new Set(chart.series.map(series => series.name));
        chart.series.forEach(series => {
            const toggle = button(series.name, () => {
                active.has(series.name) ? active.delete(series.name) : active.add(series.name);
                toggle.setAttribute('aria-pressed', String(active.has(series.name)));
                draw();
            }, 'chart-series');
            toggle.setAttribute('aria-pressed', 'true');
            const swatch = e('span', { class: 'chart-swatch', 'aria-hidden': 'true' });
            swatch.style.backgroundColor = series.color;
            toggle.insertBefore(swatch, toggle.firstChild);
            legend.append(toggle);
        });
        ui.plot.after(legend);
        function draw() {
            const [min, max] = domain(chart.domain);
            const svg = s('svg', { viewBox: '0 0 760 425', class: 'ts-svg', role: 'group', 'aria-label': `${chart.title}. ${chart.note}` });
            const x = value => 66+(value-min)/(max-min)*670, y = value => 332-value/chart.yMax*300;
            const ticks = chart.yMax > 100 ? [0,200,400,600,800,1000] : chart.yMax === 16 ? [0,2,4,6,8,10,12,14] : [0,2,4,6,8,10,12,14,16];
            ticks.forEach(v => {
                svg.append(s('line', { x1: 66, x2: 736, y1: y(v), y2: y(v), class: 'chart-gridline' }));
                text(svg, 56, y(v)+4, v.toString(), { 'text-anchor': 'end' });
            });
            for (const series of chart.series) {
                if (!active.has(series.name)) continue;
                series.bins.forEach((bin, i) => {
                    if (bin.right < min || bin.left > max) return;
                    const left = x(Math.max(min, bin.left)), right = x(Math.min(max, bin.right));
                    if (bin.count === 0) return;
                    const rect = s('rect', { x: left, y: y(bin.count), width: Math.max(0, right-left), height: 332-y(bin.count),
                        fill: series.color, 'fill-opacity': chart.series.length > 1 ? '.6' : '1', stroke: chart.series.length > 1 ? 'none' : '#111', 'stroke-width': '.7' });
                    const label = `${series.name} · bin ${i+1}: ${chart.approximateX ? 'approximately ' : ''}${number(bin.left)} to ${number(bin.right)}; ${bin.countApproximate ? `approximately ${Math.round(bin.count)} entries (image-reading range ${Math.floor(bin.low)}–${Math.ceil(bin.high)})` : `${bin.count} segments`}.`;
                    mark(rect, label, ui.status);
                    svg.append(rect);
                });
            }
            const xTicks = Number(max-min) < (chart.domain[1]-chart.domain[0]) ? Array.from({ length: 5 }, (_, i) => min+(max-min)*i/4) : chart.ticks;
            xTicks.filter(v => v >= min && v <= max).forEach(v => text(svg, x(v), 355, number(v), { 'text-anchor': 'middle' }));
            text(svg, 400, 395, chart.xLabel, { 'text-anchor': 'middle' });
            text(svg, 18, 190, id === 'trust-score' ? 'Entry count (approximate)' : 'Frequency', { transform: 'rotate(-90 18 190)', 'text-anchor': 'middle' });
            ui.plot.replaceChildren(svg);
        }
        const rows = chart.series.flatMap(series => series.bins.map((bin, i) => [series.name, i+1,
            `${chart.approximateX ? '≈' : ''}${number(bin.left)}`, `${chart.approximateX ? '≈' : ''}${number(bin.right)}`,
            `${bin.countApproximate ? '≈' : ''}${bin.countApproximate ? Math.round(bin.count) : bin.count}`,
            bin.countApproximate ? `Image-reading range: ${Math.floor(bin.low)}–${Math.ceil(bin.high)}` : 'Frequency reconstructed from figure']));
        const headers = ['Series','Bin','Lower edge','Upper edge','Frequency','Provenance / uncertainty'];
        ui.toolbar.append(button('Export figure readings ↓', () => csvDownload(`timeseries-${id}-readings.csv`, [headers, ...rows])));
        dataTable(ui.root, headers, rows);
        draw();
        mount(`ts-${id}`, ui.root);
    }
    function overlays() {
        const info = data.overlays;
        const ui = shell('Forecast and observed values by horizon',
            'Vector traces digitized from the saved plot. Hover readings describe the visible range in an image column, which can contain multiple samples. They are approximate, not recovered individual forecasts. Overlaps and image resolution limit precision; the original graphic remains below.');
        const select = e('select', { 'aria-label': 'Overlay forecast horizon' });
        for (let i = 0; i < 24; i++) select.append(e('option', { value: String(i) }, `Horizon ${i}`));
        let svg, cursor, tooltip, drag = null, cursorColumn = null, guides = true;
        const active = new Set(['Forecast', 'Observed']);
        const choose = value => { select.value = String((value+24)%24); cursorColumn = null; draw(); };
        ui.toolbar.append(button('← Previous', () => choose(Number(select.value)-1)), select, button('Next →', () => choose(Number(select.value)+1)));
        const domain = zoomControls(ui.toolbar, draw);
        domain.zoomInput.setAttribute('max', '16');
        domain.zoomInput.setAttribute('step', '.01');
        domain.positionInput.setAttribute('step', '.01');
        const legend = e('div', { class: 'chart-legend ts-overlay-legend' });
        for (const series of info.horizons[0].series) {
            const toggle = button(series.name, () => {
                active.has(series.name) ? active.delete(series.name) : active.add(series.name);
                toggle.setAttribute('aria-pressed', String(active.has(series.name)));
                draw();
            }, 'chart-series');
            toggle.setAttribute('aria-pressed', 'true');
            const swatch = e('span', { class: 'chart-swatch', 'aria-hidden': 'true' });
            swatch.style.backgroundColor = series.color;
            toggle.insertBefore(swatch, toggle.firstChild);
            legend.append(toggle);
        }
        const guidesButton = button('Break guides', () => { guides = !guides; guidesButton.setAttribute('aria-pressed', String(guides)); draw(); }, 'chart-series');
        guidesButton.setAttribute('aria-pressed', 'true');
        legend.append(guidesButton);
        const hint = e('p', { class: 'chart-hint' }, 'Hover or tap for readings. Click a legend item to hide a series. Zoom, then drag to pan. Shift + scroll zooms; arrow keys move the cursor.');
        ui.root.insertBefore(hint, ui.plot);
        ui.plot.classList.add('ts-overlay-plot');
        ui.plot.after(legend);
        select.addEventListener('change', () => choose(Number(select.value)));
        const L = 62, T = 28, W = 674, H = 294, sourceSpan = info.xDomain[1]-info.xDomain[0];
        const horizon = () => info.horizons[Number(select.value)];
        const valueAt = pixelY => horizon().yScale[0]*pixelY+horizon().yScale[1];
        const entryAt = column => info.xDomain[0]+column/info.width*sourceSpan;
        const xAt = entry => { const [min,max]=domain([0,6030]); return L+(entry-min)/(max-min)*W; };
        const pretty = v => Number(v.toFixed(3)).toString();
        function inspect(column) {
            cursorColumn = Math.max(0, Math.min(info.width-1, Math.round(column)));
            const px=xAt(entryAt(cursorColumn+.5));
            cursor.replaceChildren();
            if(px<L || px>L+W){cursor.setAttribute('visibility','hidden');tooltip.hidden=true;return;}
            cursor.setAttribute('visibility','visible');
            cursor.append(s('line',{x1:px,x2:px,y1:T,y2:T+H,stroke:'#294c3b','stroke-width':1,'stroke-dasharray':'4 4'}));
            const lines = [`Horizon ${select.value} · entries ≈${Math.round(entryAt(cursorColumn))}–${Math.round(entryAt(cursorColumn+1))}`];
            const heading=e('strong',{},lines[0]);
            tooltip.replaceChildren(heading);
            for(const series of horizon().series) {
                if(!active.has(series.name))continue;
                const range=series.ranges[cursorColumn];
                const label=range ? `${series.name}: ≈${pretty(valueAt(range[1]))}–${pretty(valueAt(range[0]))}` : `${series.name}: no visible trace`;
                lines.push(label);
                const row=e('div',{},label); row.style.color=series.color; tooltip.append(row);
                if(range) {
                    const y1=T+range[0]/info.height*H, y2=T+range[1]/info.height*H;
                    cursor.append(s('line',{x1:px,x2:px,y1,y2,stroke:series.color,'stroke-width':4,opacity:'.8'}));
                    for(const y of [y1,y2])cursor.append(s('line',{x1:px-5,x2:px+5,y1:y,y2:y,stroke:series.color,'stroke-width':2}));
                }
            }
            tooltip.append(e('small',{},'Visible trace ranges · figure estimates'));
            tooltip.style.left=px>L+W*.58?'12px':'auto';
            tooltip.style.right=px>L+W*.58?'auto':'12px';
            tooltip.hidden=false;
            ui.status.textContent=lines.join(' · ')+' · approximate readings from the saved figure.';
        }
        function draw() {
            const [min,max]=domain([0,6030]);
            svg=s('svg',{viewBox:'0 0 760 390',class:'ts-svg ts-overlay-svg',tabindex:'0',role:'group',
                'aria-label':`Horizon ${select.value}. Digitized forecast and observed traces. Left and right arrows inspect visible ranges; plus and minus zoom; Escape dismisses the cursor.`});
            const defs=s('defs'),clip=s('clipPath',{id:'ts-overlay-clip'});
            clip.append(s('rect',{x:L,y:T,width:W,height:H}));defs.append(clip);svg.append(defs);
            const lower=valueAt(info.height),upper=valueAt(0),rawStep=(upper-lower)/7;
            const magnitude=10**Math.floor(Math.log10(rawStep));
            const step=[1,2,2.5,5,10].map(v=>v*magnitude).find(v=>v>=rawStep);
            for(let value=Math.ceil(lower/step)*step;value<=upper;value+=step) {
                const y=T+(value-horizon().yScale[1])/horizon().yScale[0]/info.height*H;
                svg.append(s('line',{x1:L,x2:L+W,y1:y,y2:y,class:'chart-gridline'}));
                text(svg,L-9,y+4,pretty(value),{'text-anchor':'end'});
            }
            const clipped=s('g',{'clip-path':'url(#ts-overlay-clip)','pointer-events':'none'});
            const transform=`translate(${xAt(info.xDomain[0])} ${T}) scale(${W/(max-min)*sourceSpan/info.width} ${H/info.height})`;
            for(const series of horizon().series) {
                if(!active.has(series.name))continue;
                clipped.append(s('path',{d:series.path,transform,fill:series.color,'fill-opacity':'.7',class:'ts-trace',
                    'data-series':series.name,'aria-label':`${series.name}: digitized visible trace`}));
            }
            if(guides) {
                for(const [positions,color,dash] of [[data.predictedBreaks,'#0000ff','6 4'],[data.observedBreaks,'#ff0000','2 4']]) {
                    const px=xAt(positions[Number(select.value)]*10);
                    clipped.append(s('line',{x1:px,x2:px,y1:T,y2:T+H,stroke:color,'stroke-width':1.5,'stroke-dasharray':dash,opacity:'.65',class:'ts-break-guide'}));
                }
            }
            svg.append(clipped);
            svg.append(s('path',{d:`M${L} ${T}V${T+H}H${L+W}`,fill:'none',stroke:'#8a9e91','stroke-width':1}));
            for(let i=0;i<=6;i++)text(svg,L+W*i/6,T+H+23,String(Math.round(min+(max-min)*i/6)),{'text-anchor':'middle'});
            text(svg,L+W/2,378,'Forecast-entry order',{'text-anchor':'middle'});
            text(svg,17,T+H/2,'Value (figure reading)',{transform:`rotate(-90 17 ${T+H/2})`,'text-anchor':'middle'});
            if(!active.size)text(svg,L+W/2,T+H/2,'Select Forecast or Observed below to show a trace.',{'text-anchor':'middle'});
            cursor=s('g',{visibility:'hidden','pointer-events':'none',class:'ts-crosshair'});svg.append(cursor);
            tooltip=e('div',{class:'ts-overlay-tooltip',hidden:'','aria-hidden':'true'});
            ui.plot.replaceChildren(svg,tooltip);
            svg.addEventListener('keydown',event=>{
                if(event.key==='ArrowRight'||event.key==='ArrowLeft') {
                    event.preventDefault();
                    const midpoint=(min+max)/2, center=(midpoint-info.xDomain[0])/sourceSpan*info.width;
                    inspect((cursorColumn??center)+(event.key==='ArrowRight'?1:-1));
                } else if(event.key==='+'||event.key==='='||event.key==='-') {
                    event.preventDefault();const state=domain.state();domain.set(state.zoom*(event.key==='-'?.8:1.25),state.position);svg.focus();
                } else if(event.key==='Escape'){cursor.setAttribute('visibility','hidden');tooltip.hidden=true;}
            });
            ui.status.textContent=`Horizon ${select.value} · hover or tap the chart for visible trace ranges. Predicted break ≈${data.predictedBreaks[Number(select.value)]*10}; observed ≈${data.observedBreaks[Number(select.value)]*10} in entry order.`;
            if(cursorColumn!=null&&!drag)inspect(cursorColumn);
        }
        const pointerColumn=event=>{
            const bounds=svg.getBoundingClientRect(),px=(event.clientX-bounds.left)/bounds.width*760;
            const [min,max]=domain([0,6030]);
            return (min+(px-L)/W*(max-min)-info.xDomain[0])/sourceSpan*info.width;
        };
        ui.plot.addEventListener('pointerdown',event=>{
            inspect(pointerColumn(event));
            if(domain.state().zoom>1) {
                drag={x:event.clientX,state:domain.state(),width:svg.getBoundingClientRect().width*W/760};
                ui.plot.setPointerCapture?.(event.pointerId);
                ui.plot.classList.add('ts-is-panning');
            }
        });
        ui.plot.addEventListener('pointermove',event=>{
            if(drag) {
                const span=6030/drag.state.zoom,pan=-(event.clientX-drag.x)/drag.width*span;
                domain.set(drag.state.zoom,drag.state.position+pan/(6030-span)*100);
            } else inspect(pointerColumn(event));
        });
        const endDrag=()=>{drag=null;ui.plot.classList.remove('ts-is-panning');};
        ui.plot.addEventListener('pointerup',endDrag);
        ui.plot.addEventListener('pointercancel',endDrag);
        ui.plot.addEventListener('pointerleave',()=>{if(!drag){cursor.setAttribute('visibility','hidden');tooltip.hidden=true;}});
        ui.plot.addEventListener('wheel',event=>{
            if(!event.shiftKey)return;
            event.preventDefault();
            const column=pointerColumn(event),anchor=entryAt(column),[min,max]=domain([0,6030]);
            const scale=Math.max(1,Math.min(16,domain.state().zoom*Math.exp(-event.deltaY*.002)));
            const span=6030/scale,start=anchor-(anchor-min)/(max-min)*span;
            domain.set(scale,scale===1?0:start/(6030-span)*100);
        },{passive:false});
        draw();
        mount('ts-horizon-overlays',ui.root);
    }
    function schematic(kind) {
        const architecture = kind==='architecture';
        const ui=shell(architecture ? 'Explore the Autoformer architecture' : 'Explore Auto-Correlation',
            'Interactive schematic adapted from the credited source diagram. Select a block to read its role. Residual paths and repeated heads are simplified; the complete original diagram remains available below.');
        const svg=s('svg',{viewBox:architecture?'0 0 800 530':'0 0 760 465',class:'ts-svg ts-diagram',role:'group','aria-label':architecture?'Autoformer encoder and decoder flow':'Auto-Correlation: query and key correlation selects delays used to aggregate values'});
        const defs=s('defs'),marker=s('marker',{id:`ts-arrow-${kind}`,viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:6,markerHeight:6,orient:'auto-start-reverse'});
        marker.append(s('path',{d:'M 0 0 L 10 5 L 0 10 z',fill:'#677c70'}));defs.append(marker);svg.append(defs);
        const arrow=(d,dashed=false)=>svg.append(s('path',{d,fill:'none',stroke:'#677c70','stroke-width':1.6,'stroke-dasharray':dashed?'5 4':'','marker-end':`url(#ts-arrow-${kind})`}));
        const nodes=[];
        function node(x,y,w,label,description,color) {
            const group=s('g',{role:'button',tabindex:'0','aria-label':`${label.replaceAll('|',' ')}: ${description}`,'aria-pressed':'false',class:'ts-node'});
            group.append(s('rect',{x,y,width:w,height:58,rx:4,fill:color,stroke:'#a0afa5'}));
            label.split('|').forEach((line,i,lines)=>text(group,x+w/2,y+29+(i-(lines.length-1)/2)*18+5,line,{'text-anchor':'middle',class:'ts-node-label'}));
            const select=()=>{nodes.forEach(n=>n.setAttribute('aria-pressed',String(n===group)));ui.status.textContent=`${label.replaceAll('|',' ')} — ${description}`;};
            group.addEventListener('click',select);group.addEventListener('focus',select);
            group.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();select();}});
            nodes.push(group);svg.append(group);
        }
        const green='#c6e2b0',blue='#bce4f5',pink='#f6dbe0',gold='#fff0b2';
        const decomp='A moving average extracts a trend component; subtraction leaves the seasonal remainder. This remainder may also contain short fluctuations and noise.';
        if(architecture) {
            svg.append(s('rect',{x:10,y:15,width:780,height:185,rx:10,fill:'#fafbf8',stroke:'#b7c3ba'}));
            svg.append(s('rect',{x:10,y:270,width:780,height:240,rx:10,fill:'#fafbf8',stroke:'#b7c3ba'}));
            text(svg,28,43,'Encoder · repeat N times',{class:'ts-svg-heading'});
            text(svg,28,296,'Decoder · repeat M times',{class:'ts-svg-heading'});
            [165,360,555].forEach(x=>arrow(`M${x} 119 H${x+65}`));
            node(35,90,130,'Auto-|Correlation','Encoder queries, keys, and values identify recurring delays and aggregate representations across those delays.',green);
            node(230,90,130,'Series|decomposition',decomp,blue);
            node(425,90,130,'Feed|forward','A position-wise feed-forward transformation processes the seasonal representation, with a residual connection in the original block.',pink);
            node(620,90,130,'Series|decomposition',`${decomp} The encoder passes its seasonal representation to the decoder.`,blue);
            arrow('M685 148 V240 H324 V335',true);
            text(svg,440,230,'Encoder seasonal memory',{'text-anchor':'middle'});
            const xs=[15,145,275,405,535,665];
            xs.slice(0,-1).forEach(x=>arrow(`M${x+98} 364 H${x+130}`));
            node(15,335,98,'Auto-|Correlation','The initialized seasonal decoder input is processed through a self Auto-Correlation block.',green);
            node(145,335,98,'Series|decomp',decomp,blue);
            node(275,335,98,'Cross Auto-|Correlation','Decoder queries use the encoder seasonal memory as keys and values, connecting historical patterns to the forecast representation.',green);
            node(405,335,98,'Series|decomp',decomp,blue);
            node(535,335,98,'Feed|forward','The decoder applies a feed-forward transformation before another decomposition.',pink);
            node(665,335,98,'Series|decomp',decomp,blue);
            [194,454,714].forEach(x=>arrow(`M${x} 393 V455`));
            arrow('M35 455 H775');
            text(svg,35,435,'Trend initialization');
            text(svg,365,485,'Accumulated trend components',{'text-anchor':'middle'});
            arrow('M763 364 H775 V455');
            svg.append(s('circle',{cx:775,cy:455,r:12,fill:'white',stroke:'#677c70'}));
            text(svg,775,461,'+',{'text-anchor':'middle',class:'ts-node-label'});
            text(svg,775,495,'Forecast',{'text-anchor':'end'});
            text(svg,28,175,'Seasonal input → recurring delays → decomposition → seasonal representation');
        } else {
            arrow('M95 365 V338');arrow('M265 365 V338');arrow('M605 365 V208');
            arrow('M95 280 V249 H205 V228');arrow('M265 280 V249 H205 V228');
            arrow('M205 170 V148');arrow('M265 119 H355');arrow('M485 119 H520 V179 H545');
            arrow('M605 150 V133');arrow('M605 75 V63');
            node(35,365,120,'Linear · Q','Project the query sequence. Its length supplies the target length for correlation.', '#e7eee7');
            node(205,365,120,'Linear · K','Project and resize the key sequence to the query length before computing its transform.', '#e7eee7');
            node(545,365,120,'Linear · V','Project and resize the value sequence. Values supply the representations combined by time-delay aggregation.', '#e7eee7');
            node(35,280,120,'FFT · Q','Transform queries to the frequency domain.',green);
            node(205,280,120,'FFT · K','Transform keys to the frequency domain.',green);
            node(145,170,120,'Multiply','Multiply the query transform by the complex conjugate of the key transform.',blue);
            node(145,90,120,'Inverse FFT','Transform the product back to obtain correlation scores over delays.',green);
            node(355,90,130,'Top-k|delays','Select strong delay scores and normalize the selected weights.',gold);
            node(545,150,120,'Time-delay|aggregation','Shift the value sequence by the selected delays and combine the shifted representations using normalized weights.',blue);
            node(545,75,120,'Concatenate','Concatenate the parallel head outputs.',gold);
            node(545,5,120,'Linear output','Project the concatenated representations to the output space.', '#e7eee7');
            text(svg,95,450,'Queries',{'text-anchor':'middle'});text(svg,265,450,'Keys',{'text-anchor':'middle'});text(svg,605,450,'Values',{'text-anchor':'middle'});
        }
        let current=-1;
        const step=direction=>{current=(current+direction+nodes.length)%nodes.length;nodes[current].focus();};
        ui.toolbar.append(button('← Previous block',()=>step(-1)),button('Next block →',()=>step(1)));
        ui.plot.append(svg);
        ui.status.textContent='Select a colored block, or use Next block, to explore the computation.';
        mount(`ts-${kind}`,ui.root);
    }
    // Construct each figure independently; a failure leaves its original visible.
    const tasks=[()=>schematic('architecture'),()=>schematic('autocorrelation'),heatmap,overlays,
        ...Object.entries(data.histograms).map(([id,chart])=>()=>histogram(id,chart))];
    tasks.forEach(task=>{try{task();}catch(error){console.error('Retaining original time-series graphic.',error);}});
})();
