var app = angular.module('anchorApp', []);

app.controller('anchorController', function($scope, $timeout, $http, $window) {
  var ctrl = this;

  // Main data container for the app. Stores a list of anchor object.
  // Each anchor object includes anchors and an associated topic.
  ctrl.anchors = [];
  
  ctrl.anchorUndo = [];
  
  ctrl.anchorRedo = [];

  // Since finished toggles the thank you screen, done both sends the anchor
  // data to the server and terminates the interactive session.
  ctrl.finished = false;
  ctrl.done = function() {
    if(window.confirm("Are you sure you are done?")) {
      var data = JSON.stringify(ctrl.anchors)
      $http.post("/finished", data).success(function(data, status) {
        ctrl.finished = true;
      });
    }
  };

  // Gets the vocabularly for autocomplete and anchor word validation.
  ctrl.vocab = [];
  $.get("/vocab", function(data) {
    ctrl.vocab = data.vocab;
  });

  // Creates and appends an empty anchor the anchors list.
  ctrl.addAnchor = function() {
    var anchorObj = {"anchors":[], "topic":[]};
    ctrl.anchors.push(anchorObj);
  }

  // Removes an anchor from the anchors list.
  ctrl.removeAnchor = function(index) {
    ctrl.anchors.splice(index, 1);
    $('#undoForm').removeClass('unchanged'); 
    $('#undoForm button').prop('disabled',false);
  }

  // Adds an anchor word to an existing anchor.
  ctrl.addAnchorWord = function(textForm, newAnchor) {
    $scope.$broadcast("autofillfix:update");
    var lowercaseAnchor = textForm.target.children[0].value.toLowerCase();

    var inVocab = false;
    for (var i = 0; i < ctrl.vocab.length; i++) {
      if (ctrl.vocab[i] === lowercaseAnchor) {
        inVocab = true;
        break;
      }
    }

    if (inVocab) {
      newAnchor.push(lowercaseAnchor);
        // Timeout ensures the anchor is added before the popover appears.
        $timeout(function() {
          $(".updateTopicsButtonClean").popover({
            placement:'top',
            trigger:'manual',
            html:true,
            content:'To see topic words for new anchors, press "Update Topics" here.'
          }).popover('show')
          .addClass("updateTopicsButtonDirty")
            .removeClass("updateTopicsButtonClean");
            // Indicates how long the popover stays visible.
            $timeout(function() {
              $(".updateTopicsButtonDirty").popover('hide')
                .addClass("updateTopicsButtonClean")
                .removeClass("updateTopicsButtonDirty");
            }, 5000);
        }, 20);
      textForm.target.children[0].value = "";
    } else {
      angular.element(textForm.target).popover({
        placement:'bottom',
        trigger:'manual',
        html:true,
        content:'Invalid anchor word.'
      }).popover('show');
      $timeout(function() {
        angular.element(textForm.target).popover('hide')
      }, 2000);
    }
    $('#undoForm').removeClass('unchanged');
    $('#undoForm button').prop('disabled',false);
  }

  // Deletes a word from an existing anchor.
  ctrl.deleteWord = function(closeButton, array) {
    var toClose = closeButton.target.parentNode.id;
    $("#"+toClose).remove();
    var wordIndex = array.indexOf(closeButton.target.parentNode.textContent.replace(/✖/, "").replace(/\s/g, ''));
    if (wordIndex !== -1) {
      array.splice(wordIndex, 1);
    }
    $('#undoForm').removeClass('unchanged'); 
    $('#undoForm button').prop('disabled',false);
  }

  ctrl.getNewTopics = function() {
    var currentAnchors = [];
    $(".anchorContainer").each(function() {
      // Parse out the comma-separated anchors from all the html
      var value = $(this).html().replace(/\s/g, '').replace(/<span[^>]*>/g, '').replace(/<\/span><\/span>/g, ',');
        value = value.replace(/<!--[^>]*>/g, '').replace(/,$/, '').replace(/,$/, '').replace(/\u2716/g, '');
        value = value.replace(/\&lt;/g, '<').replace(/\&gt;/g, '>');
        if (value !== "") {
          currentAnchors.push(value.split(","));
        }
    })

    ctrl.loading = true
    if (currentAnchors.length !== 0) {
      $.get("/topics", {anchors: JSON.stringify(currentAnchors)}, function(data) {
        $('#undoForm').addClass('unchanged'); 
        ctrl.anchorUndo.push(data);
        ctrl.displayNewAnchors(data);
      }).fail(function(){
        $window.alert("Update Failed. Try Clicking Update Topics Again.");
      }).always(function(){
        ctrl.loading = false;
        $scope.$apply();     
      });
    } else {
      $.get("/topics", function(data) {
        $('#undoForm').addClass('unchanged');
        ctrl.anchorUndo.push(data);
        ctrl.displayNewAnchors(data);
      }).fail(function(){
        $window.alert("Update Failed. Try Clicking Update Topics Again.");
      }).always(function(){
        ctrl.loading = false;
        $scope.$apply();      
      })
    }
  }
  ctrl.getNewTopics();

  ctrl.setAccuracy = function setAccuracy(accuracy) {
    ctrl.classifierAccuracy = accuracy
    $('#accuracyHolder').text('Accuracy: ' + (accuracy*100).toFixed(2) + '%');
  };
  
  ctrl.displayNewAnchors = function displayNewAnchors(data) {
    ctrl.anchors = getAnchorsArray(data["anchors"], data["topics"]);
    ctrl.setAccuracy(data['accuracy']);
    $(".top-to-bottom").css("height", $(".anchors-and-topics").height());
  };
  
  ctrl.undoAction = function undoAction() {
    if($('#undoForm.unchanged').length > 0){
      lastModified = ctrl.anchorUndo.pop();
      if(ctrl.anchorUndo.length < 1) {
        ctrl.anchorUndo.push(lastModified);
      }
    }
    if(ctrl.anchorUndo.length > 0) {
      lastModified = ctrl.anchorUndo.pop();
      ctrl.displayNewAnchors(lastModified);
      ctrl.anchorUndo.push(lastModified);
    }
    $('#undoForm').addClass('unchanged');
  };

});

app.directive("autofillfix", function() {
  //This is required because of some problem between Angular and autofill
  return {
    require: "ngModel",
    link: function(scope, element, attrs, ngModel) {
      scope.$on("autofillfix:update", function() {
      ngModel.$setViewValue(element.val());
      })
    }
  }
});

app.directive("autocomplete", function() {
  return {
    restrict: 'A',
    link: function(scope, elem, attr, ctrl) {
      elem.autocomplete({
        source: scope.ctrl.vocab,
        minLength: 2,
        // This function is called whenever a list choice is selected
        select: function(event, ui) {
          // This sets a listener to prevent the page from reloading
          $(this).parents("form").on('submit', function() {
            return false;
          });
          // This moves the selected value into the input before the
          //   input is submitted
          $(this).val(ui.item.value);
          // This triggers the submit event, which turns the selected
          //   word into a proper anchor word (with the border)
          $(this).parents("form").submit();
          // This prevents the value from being duplicated
          return false;
        }
      }).keypress(function(e) {
        // This closes the menu when the enter key is pressed
        if (!e) e = window.event;
        if (e.keyCode == '13') {
          $(".anchorInput" ).autocomplete('close');
          // This sets a listener to prevent the page from reloading
          $(this).parents("form").on('submit', function() {
            return false;
          });
          // This triggers the submit event, which turns the selected
          //   word into a proper anchor word (with the border)
          $(this).parents("form").submit();
          return false;
        }
      })
    }
  }
});

// Converts separate anchor and topic word lists into an array of anchor objects.
var getAnchorsArray = function(anchors, topics) {
    var result = []
    for (var i = 0; i < anchors.length; i++) {
        result.push({"anchors": anchors[i], "topic": topics[i]});
    }
    return result
};

//All functions below here enable dragging and dropping

var allowDrop = function(ev) {
  ev.preventDefault();
};

var drag = function(ev) {
  ev.dataTransfer.setData("text", ev.target.id);
};

//Holds next id for when we copy nodes
var copyId = 0;

var drop = function(ev) {
  ev.preventDefault()
  var data = ev.dataTransfer.getData("text");
  var dataString = JSON.stringify(data);
  //If an anchor or a copy of a topic word, drop
  if (dataString.indexOf("anchor") !== -1 || dataString.indexOf("copy") !== -1) {
    //Need to cover all the possible places in the main div it could be dropped
    if($(ev.target).hasClass( "droppable" )) {
        ev.target.appendChild(document.getElementById(data));
    }
    else if($(ev.target).hasClass( "draggable" )) {
        $(ev.target).parent()[0].appendChild(document.getElementById(data));
    }
    else if($(ev.target).hasClass( "anchorInputContainer" )) {
        $(ev.target).siblings(".anchorContainer")[0].appendChild(document.getElementById(data));
    }
    else if ($(ev.target).hasClass( "anchorInput" )) {
        $(ev.target).parent().parent().siblings(".anchorContainer")[0].appendChild(document.getElementById(data));
    }
    else if ($(ev.target).hasClass( "anchor" )) {
        $(ev.target).children(".anchorContainer")[0].appendChild(document.getElementById(data));
    }
  }
  //If a topic word, copy it
  else {
    var nodeCopy = document.getElementById(data).cloneNode(true);
    nodeCopy.id = data + "copy" + copyId++;
    var closeButton = addDeleteButton(nodeCopy.id + "close");
    nodeCopy.appendChild(closeButton);
    //Need to cover all the possible places in the main div it could be dropped
    if($(ev.target).hasClass( "droppable" )) {
        ev.target.appendChild(nodeCopy);
    }
    else if($(ev.target).hasClass( "draggable" )) {
        $(ev.target).parent()[0].appendChild(nodeCopy);
    }
    else if($(ev.target).hasClass( "anchorInputContainer" )) {
        $(ev.target).siblings(".anchorContainer")[0].appendChild(nodeCopy);
    }
    else if ($(ev.target).hasClass( "anchorInput" )) {
        $(ev.target).parent().parent().siblings(".anchorContainer")[0].appendChild(nodeCopy);
    }
    else if ($(ev.target).hasClass( "anchor" )) {
        $(ev.target).children(".anchorContainer")[0].appendChild(nodeCopy);
    }
  }
  $('#undoForm').removeClass('unchanged');
  $('#undoForm button').prop('disabled',false);
};

//used to delete words that are copies (because they can't access the function in the Angular scope)
var deleteWord = function(ev) {
  $("#"+ev.target.id).parent()[0].remove();
  $('#undoForm').removeClass('unchanged');
  $('#undoForm button').prop('disabled',false);
};

//Adds a delete button (little 'x' on the right side) of an anchor word
var addDeleteButton = function(id) {
  var closeButton = document.createElement("span");
  closeButton.innerHTML = " &#10006";
  var closeClass = document.createAttribute("class");
  closeClass.value = "close";
  closeButton.setAttributeNode(closeClass);
  var closeId = document.createAttribute("id");
  closeId.value = id;
  closeButton.setAttributeNode(closeId);
  var closeClick = document.createAttribute("onclick");
  closeClick.value = "deleteWord(event)";
  closeButton.setAttributeNode(closeClick);
  return closeButton;
};
