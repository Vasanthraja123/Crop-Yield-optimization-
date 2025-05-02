document.addEventListener('DOMContentLoaded', function () {
  // Function to trigger Google Translate language change
  function changeLanguage(lang) {
    var select = document.querySelector('select.goog-te-combo');
    if (!select) {
      console.warn('Google Translate select element not found.');
      return;
    }
    select.value = lang;
    // Trigger change event
    var event = document.createEvent('HTMLEvents');
    event.initEvent('change', true, true);
    select.dispatchEvent(event);
  }

  // Add click event listeners to language options
  var languageOptions = document.querySelectorAll('.language-option');
  languageOptions.forEach(function (option) {
    option.addEventListener('click', function (e) {
      e.preventDefault();
      var lang = this.getAttribute('data-lang');
      changeLanguage(lang);
    });
  });
});
