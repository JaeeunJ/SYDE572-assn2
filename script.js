document.addEventListener("DOMContentLoaded", function() {
    var defaultTab = document.getElementById("defaultOpen");
    defaultTab.click();
    openPart(event, "part1");

    defaultTab.className += " active";
});

function openPart(evt, partNo) {
    var i, bentoContainer, tablinks;
  
    bentoContainer = document.getElementsByClassName("bento-container");
    for (i = 0; i < bentoContainer.length; i++) {
      bentoContainer[i].style.display = "none";
    }

    tablinks = document.getElementsByClassName("tablinks");
    for (i = 0; i < tablinks.length; i++) {
      tablinks[i].className = tablinks[i].className.replace(" active", "");
    }
  
    document.getElementById(partNo).style.display = "grid";
    evt.currentTarget.className += " active";
}