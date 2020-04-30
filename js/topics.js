function addTopics() {
  var select = document.getElementById("topic_selection");
  $.getJSON("../static/json/topics.json", function(topics) {
    for (var i = 0; i < topics.length; i++) {
      var option = document.createElement("option");
      var curr = topics[i]

      option.value = curr[0]
      option.text = curr[1]
      select.add(option);
    }
  })
};
