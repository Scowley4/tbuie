var app = angular.module('anchorApp', [])
app.controller('anchorController', function($scope, $timeout, $http) {
  var ctrl = this

  // Main data container for the app. Stores a list of anchor object.
  // Each anchor object includes anchors and an associated topic.
  ctrl.anchors = []

  // Since finished toggles the thank you screen, done both sends the anchor
  // data to the server and terminates the interactive session.
  ctrl.finished = false
  ctrl.done = function() {
    if(window.confirm("Are you sure you are done?")) {
      var data = JSON.stringify(ctrl.anchors)
      $http.post("/finished", data).success(function(data, status) {
        ctrl.finished = true
      })
    }
  }

  // Gets the vocabularly for autocomplete and anchor word validation.
  ctrl.vocab = []
  $.get("/vocab", function(data) {
    ctrl.vocab = data.vocab
  })

  // Creates and appends an empty anchor the anchors list.
  ctrl.addAnchor = function() {
    var anchorObj = {"anchors":[], "topic":[]}
    ctrl.anchors.push(anchorObj)
  }

  // Removes an anchor from the anchors list.
  ctrl.removeAnchor = function(index) {
    ctrl.anchors.splice(index, 1)
  }

  // Adds an anchor word to an existing anchor.
  ctrl.addAnchorWord = function(textForm, newAnchor) {
    $scope.$broadcast("autofillfix:update")
    var lowercaseAnchor = textForm.target.children[0].value.toLowerCase()

    var inVocab = false
    for (var i = 0; i < ctrl.vocab.length; i++) {
      if (ctrl.vocab[i] === lowercaseAnchor) {
        inVocab = true
        break
      }
    }

    if (inVocab) {
      newAnchor.push(lowercaseAnchor)
        // Timeout ensures the anchor is added before the popover appears.
        $timeout(function() {
          $(".updateTopicsButtonClean").popover({
            placement:'top',
            trigger:'manual',
            html:true,
            content:'To see topic words for new anchors, press "Update Topics" here.'
          }).popover('show')
          .addClass("updateTopicsButtonDirty")
            .removeClass("updateTopicsButtonClean")
            // Indicates how long the popover stays visible.
            $timeout(function() {
              $(".updateTopicsButtonDirty").popover('hide')
                .addClass("updateTopicsButtonClean")
                .removeClass("updateTopicsButtonDirty")
            }, 5000)
        }, 20)
      textForm.target.children[0].value = ""
    } else {
      angular.element(textForm.target).popover({
        placement:'bottom',
        trigger:'manual',
        html:true,
        content:'Invalid anchor word.'
      }).popover('show')
      $timeout(function() {
        angular.element(textForm.target).popover('hide')
      }, 2000)
    }
  }

  // Deletes a word from an existing anchor.
  ctrl.deleteWord = function(closeButton, array) {
    var toClose = closeButton.target.parentNode.id
    $("#"+toClose).remove()
    var wordIndex = array.indexOf(closeButton.target.parentNode.textContent.replace(/✖/, "").replace(/\s/g, ''))
    if (wordIndex !== -1) {
      array.splice(wordIndex, 1)
    }
  }

  // This function only gets the topics when we have no current anchors.
  ctrl.getTopics = function() {
    ctrl.loading = true

    $.get("/topics", function(data) {
      ctrl.anchors = getAnchorsArray(data["anchors"], data["topics"])
      ctrl.setAccuracy(data['accuracy'])
      ctrl.loading = false
      $scope.$apply()
      $(".top-to-bottom").css("height", $(".anchors-and-topics").height())
    })
  }
  ctrl.getTopics()

  //This function takes all anchors from the left column and gets their new topic words.
  //  It then updates the page to include the new topic words.
  ctrl.getNewTopics = function() {
      var currentAnchors = []
      if ($(".anchorContainer").length !== 0) {
          //If needed, this checks if the anchors all only have 1 word
          $(".anchorContainer").each(function() {
              //This parses out just the comma-separated anchors from all the html
              var value = $(this).html().replace(/\s/g, '').replace(/<span[^>]*>/g, '').replace(/<\/span><\/span>/g, ',')
              value = value.replace(/<!--[^>]*>/g, '').replace(/,$/, '').replace(/,$/, '').replace(/\u2716/g, '')
              //This prevents errors on the server if there are '<' or '>' symbols in the anchors
              value = value.replace(/\&lt;/g, '<').replace(/\&gt;/g, '>')
              if (value === "") {
                  return true
              }
              var tempArray = value.split(",")
              currentAnchors.push(tempArray)
          })

          if (currentAnchors.length !== 0) {

              var getParams = JSON.stringify(currentAnchors)

              ctrl.loading = true

              $.get("/topics", {anchors: getParams}, function(data) {
                  var saveState = {anchors: currentAnchors,
                                   topics: data["topics"]}
                  //Update the anchors in the UI
                  ctrl.anchors = getAnchorsArray(currentAnchors, data["topics"])
                  ctrl.setAccuracy(data['accuracy'])
                  ctrl.loading = false
                  $scope.$apply()
                  // Sets the height of the document container
                  $(".top-to-bottom").css("height", $(".anchors-and-topics").height())
              })
          } else {
              ctrl.getTopics()
          }
      } else {
          ctrl.getTopics()
      }
  }


        // This sets the height of the document container on load
        $timeout(function() {
          $(".top-to-bottom").css("height", $(".anchors-and-topics").height())
        }, 50)


        ctrl.setAccuracy = function setAccuracy(accuracy) {
          if (!accuracy) {
            $('#accuracyHolder').text('No accuracy yet')
          }
          else {
            ctrl.classifierAccuracy = accuracy
            $('#accuracyHolder').text('Accuracy: ' + (accuracy*100).toFixed(2) + '%')
          }
        }

        ctrl.showSampleDocuments = false

        ctrl.topicDocuments = []


        ctrl.popoverIfDisabled = function(index) {
            var selector = "#show-docs-button-" + index
            var btn = $(selector)
            var disabled = btn.prop('disabled')
            var anyOpen = false
            $("[id^=show-docs-button-]").each(function() {
                var pop = $(this).parent().data('bs.popover')
                if (pop !== undefined)
                {
                    anyOpen = pop.tip().hasClass('in')
                }
            })
            btn.popover()
            if (disabled && !anyOpen) {
                var parent = btn.parent()
                parent.popover({
                    placement: 'bottom',
                    trigger: 'manual',
                    html: true,
                    content: 'Click "Update Topics" to sample new documents.'
                }).popover('show')
                $timeout(function() {
                    ctrl.closePopover(index)
                }, 3000)
            }
        }

      ctrl.closePopover = function closePopover(index) {
          var selector = "#show-docs-button-" + index
          var coke = $(selector)
          coke.parent().popover('destroy')
      }


    }).directive("autofillfix", function() {
        //This is required because of some problem between Angular and autofill
        return {
            require: "ngModel",
            link: function(scope, element, attrs, ngModel) {
                scope.$on("autofillfix:update", function() {
                    ngModel.$setViewValue(element.val())
                })
            }
        }
    })

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
            return false
          })
          // This moves the selected value into the input before the
          //   input is submitted
          $(this).val(ui.item.value)
          // This triggers the submit event, which turns the selected
          //   word into a proper anchor word (with the border)
          $(this).parents("form").submit()
          // This prevents the value from being duplicated
          return false
        }
      }).keypress(function(e) {
        // This closes the menu when the enter key is pressed
        if (!e) e = window.event
        if (e.keyCode == '13') {
          $(".anchorInput" ).autocomplete('close')
          // This sets a listener to prevent the page from reloading
          $(this).parents("form").on('submit', function() {
            return false
          })
          // This triggers the submit event, which turns the selected
          //   word into a proper anchor word (with the border)
          $(this).parents("form").submit()
          return false
        }
      })
    }
  }
})

// Converts separate anchor and topic word lists into an array of anchor objects.
var getAnchorsArray = function(anchors, topics) {
    var result = []
    for (var i = 0; i < anchors.length; i++) {
        result.push({"anchors": anchors[i], "topic": topics[i]})
    }
    return result
}

//All functions below here enable dragging and dropping
//They could possibly be in another file and included?


var allowDrop = function(ev) {
    ev.preventDefault()
}


var drag = function(ev) {
    ev.dataTransfer.setData("text", ev.target.id)
}


//Holds next id for when we copy nodes
var copyId = 0


var drop = function(ev) {
    ev.preventDefault()
    var data = ev.dataTransfer.getData("text")
    var dataString = JSON.stringify(data)
    //If an anchor or a copy of a topic word, drop
    if (dataString.indexOf("anchor") !== -1 || dataString.indexOf("copy") !== -1) {
        //Need to cover all the possible places in the main div it could be dropped
        if($(ev.target).hasClass( "droppable" )) {
            ev.target.appendChild(document.getElementById(data))
        }
        else if($(ev.target).hasClass( "draggable" )) {
            $(ev.target).parent()[0].appendChild(document.getElementById(data))
        }
        else if($(ev.target).hasClass( "anchorInputContainer" )) {
            $(ev.target).siblings(".anchorContainer")[0].appendChild(document.getElementById(data))
        }
        else if ($(ev.target).hasClass( "anchorInput" )) {
            $(ev.target).parent().parent().siblings(".anchorContainer")[0].appendChild(document.getElementById(data))
        }
        else if ($(ev.target).hasClass( "anchor" )) {
            $(ev.target).children(".anchorContainer")[0].appendChild(document.getElementById(data))
        }
    }
    //If a topic word, copy it
    else {
        var nodeCopy = document.getElementById(data).cloneNode(true)
        nodeCopy.id = data + "copy" + copyId++
        var closeButton = addDeleteButton(nodeCopy.id + "close")
        nodeCopy.appendChild(closeButton)
        //Need to cover all the possible places in the main div it could be dropped
        if($(ev.target).hasClass( "droppable" )) {
            ev.target.appendChild(nodeCopy)
        }
        else if($(ev.target).hasClass( "draggable" )) {
            $(ev.target).parent()[0].appendChild(nodeCopy)
        }
        else if($(ev.target).hasClass( "anchorInputContainer" )) {
            $(ev.target).siblings(".anchorContainer")[0].appendChild(nodeCopy)
        }
        else if ($(ev.target).hasClass( "anchorInput" )) {
            $(ev.target).parent().parent().siblings(".anchorContainer")[0].appendChild(nodeCopy)
        }
        else if ($(ev.target).hasClass( "anchor" )) {
            $(ev.target).children(".anchorContainer")[0].appendChild(nodeCopy)
        }
    }
}


//used to delete words that are copies (because they can't access the function in the Angular scope)
var deleteWord = function(ev) {
    $("#"+ev.target.id).parent()[0].remove()
}


//Adds a delete button (little 'x' on the right side) of an anchor word
var addDeleteButton = function(id) {
    var closeButton = document.createElement("span")
    closeButton.innerHTML = " &#10006"
    var closeClass = document.createAttribute("class")
    closeClass.value = "close"
    closeButton.setAttributeNode(closeClass)
    var closeId = document.createAttribute("id")
    closeId.value = id
    closeButton.setAttributeNode(closeId)
    var closeClick = document.createAttribute("onclick")
    closeClick.value = "deleteWord(event)"
    closeButton.setAttributeNode(closeClick)
    return closeButton
}
