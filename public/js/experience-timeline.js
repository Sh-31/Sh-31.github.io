document.addEventListener('DOMContentLoaded', function () {
    const timelineItems = document.querySelectorAll('.timeline-item');

    // Arrow function 'entries =>' converted to 'function (entries)'
    const observer = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
            if (entry.isIntersecting) {
                entry.target.classList.add('is-visible');
            }
        });
    }, {
        // Line 17 is likely here (or just before)
        // Ensure no other arrow functions are present
        threshold: 0.1
    });

    timelineItems.forEach(function (item) {
        observer.observe(item);
    });
});