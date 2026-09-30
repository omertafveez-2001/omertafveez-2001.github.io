const figureDialog = document.querySelector('.figure-dialog');

if (figureDialog && typeof figureDialog.showModal === 'function') {
    const image = figureDialog.querySelector('.dialog-image');
    const title = figureDialog.querySelector('#figure-dialog-title');
    const original = figureDialog.querySelector('.dialog-original');

    document.querySelectorAll('[data-figure]').forEach(link => {
        link.addEventListener('click', event => {
            if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
            event.preventDefault();
            const thumbnail = link.querySelector('img');
            image.src = link.href;
            image.alt = thumbnail.alt;
            image.style.setProperty('--figure-width', `${thumbnail.naturalWidth || 1000}px`);
            title.textContent = link.dataset.figure;
            original.href = link.href;
            figureDialog.showModal();
            figureDialog.querySelector('.dialog-image-wrap').scrollTo(0, 0);
        });
    });

    figureDialog.querySelector('.dialog-close').addEventListener('click', () => figureDialog.close());
    figureDialog.addEventListener('click', event => {
        if (event.target !== figureDialog) return;
        const rect = figureDialog.getBoundingClientRect();
        if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) {
            figureDialog.close();
        }
    });
}

if ('IntersectionObserver' in window) {
    const links = [...document.querySelectorAll('.project-sidebar a')];
    const observer = new IntersectionObserver(entries => {
        for (const entry of entries) {
            if (!entry.isIntersecting) continue;
            links.forEach(link => {
                if (link.hash === '#' + entry.target.id) link.setAttribute('aria-current', 'location');
                else link.removeAttribute('aria-current');
            });
        }
    }, { rootMargin: '-5% 0px -65% 0px' });
    document.querySelectorAll('.article-section').forEach(section => observer.observe(section));
}
