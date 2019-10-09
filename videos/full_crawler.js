// [Lol Esports VOD youtube page](https://www.youtube.com/channel/UCzAypSoOFKCZUts3ULtVT_g/videos)에서 풀영상을 크롤링
// 2019 lck spring full https://www.youtube.com/watch?v=t9xmUeh_l-c&list=PLcvpEVobSi9doLKwB08V8o9l80clUgFVP
// 2019 lck summer full https://www.youtube.com/watch?v=Mh2y8MMYEms&list=PLcvpEVobSi9d3UzPrI1N-3rTI4pn11r5L

var teamNames = {
    'sb': 'sb',
    'sbg': 'sb',
    'kt': 'kt',
    'af': 'af',
    'afs': 'af',
    'grf': 'grf',
    'dwg': 'dwg',
    'dmw': 'dwg',
    'hle': 'hle',
    'sk': 'skt',
    'skt': 'skt',
    'gen': 'gen',
    'kz': 'kz',
    'jag': 'jag'
}
var videos = {}

function getTeam(teamNames, title) {
    var regexResults = /([a-zA-Z]*) vs ([a-zA-Z]*)/.exec(title)
    try {
        var team1 = teamNames[regexResults[1]]
        var team2 = teamNames[regexResults[2]]
        if (team1 != null && team2 != null) {
            return [team1, team2]
        }
    } catch (e) {
    }
    return null
}

function videoURLs() {
    var nov = {}    // 매치를 구분하기 위해 number of video 를 저장한다.
    var items = document.getElementsByTagName('ytd-playlist-panel-video-renderer')
    var urls = {};
    for (element of items) {
        for (span of element.getElementsByTagName("span")) {
            if (span.id == "video-title") {
                var title = span.title.replace(/\./gi,"").toLowerCase()
                var team = getTeam(teamNames, title)
                if (team == null) {
                    continue
                }
                var game = /game ([0-9])/.exec(title)
                if (game == null || game.length < 2) {
                    continue
                }

                var week = /week ([0-9]*)/.exec(title)
                var matchName = "lckspringsplit_" + team[0] + "_" + team[1]
                // var matchName = "lcksummersplit_" + team[0] + "_" + team[1]
                if (week != null && week.length > 1) {
                    matchName += "_week_" + week[1] + "_game_" + game[1]
                } else if (title.includes("finals")) {
                    matchName += "_finals_game_" + game[1]
                } else if (title.includes("playoffs") || title.includes("playoff")) {
                    matchName += "_playoffs_game_" + game[1]
                } else if (title.includes("wildcard") || title.includes("wildcards")) {
                    matchName += "_wildcards_game_" + game[1]
                } else if (title.includes("regional qualifier")) {
                    matchName = "regional_qualifier_" + team[0] + "_" + team[1] + "_game_" + game[1]
                } else {
                    continue
                }
                for (atag of element.getElementsByTagName("a")) {
                    if (atag.id == "wc-endpoint") {
                        url = atag.href
                        break
                    }
                }
                urls[matchName] = url;
                break
            }
        }
    }
    return urls;
}

function downloadObjectAsJson(exportObj, exportName) {
    var dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(exportObj));
    var downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", exportName + ".json");
    document.body.appendChild(downloadAnchorNode); // required for firefox
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
}

videos = videoURLs()
