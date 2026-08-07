/*
	flowpick.js — 오버레이 패널 라우터 (바닐라 JS, 의존성 없음)

	중앙 명함 헤더와 아티클 패널 사이를 해시 라우팅으로 전환한다.
	해시가 유효한 패널 id이면 그 패널을 열고, 없으면 홈 상태를 유지한다.
	열린 동안 포커스를 패널 안에 가두고, 닫으면 열었던 링크로 되돌린다.
*/
(function () {
	'use strict';

	var body = document.body;
	var card = document.getElementById('card');
	var panelsRoot = document.getElementById('panels');
	if (!panelsRoot) return; // 패널 컨테이너가 없으면 라우터가 동작할 대상이 없다

	var navLinks = Array.prototype.slice.call(document.querySelectorAll('.navlist__link'));
	var panelEls = Array.prototype.slice.call(panelsRoot.querySelectorAll('.panel'));
	var panelMap = {};
	panelEls.forEach(function (p) { panelMap[p.id] = p; });

	var reduceMotion = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
	var TRANSITION_FALLBACK = 400; // transitionend가 안 걸리는 요소/브라우저 대비 폴백(ms)
	var FOCUSABLE_SELECTOR = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

	var activePanel = null;
	var lastTrigger = null; // 닫을 때 포커스를 되돌릴 대상(연 내비 링크)

	function panelIdFromHash() {
		var id = location.hash.slice(1);
		return id && panelMap[id] ? id : null;
	}

	function linkForId(id) {
		return navLinks.filter(function (l) {
			return l.getAttribute('href') === '#' + id;
		})[0] || null;
	}

	function getFocusable(container) {
		return Array.prototype.slice.call(container.querySelectorAll(FOCUSABLE_SELECTOR));
	}

	function onKeydown(e) {
		if (!activePanel) return;
		if (e.key === 'Escape' || e.key === 'Esc') {
			goHome();
			return;
		}
		if (e.key !== 'Tab') return;
		// 포커스 트랩: 패널 안의 첫/마지막 요소 경계에서만 순환시킨다
		var focusable = getFocusable(activePanel);
		if (!focusable.length) return;
		var first = focusable[0];
		var last = focusable[focusable.length - 1];
		if (e.shiftKey && document.activeElement === first) {
			e.preventDefault();
			last.focus();
		} else if (!e.shiftKey && document.activeElement === last) {
			e.preventDefault();
			first.focus();
		}
	}

	function onOutsideClick(e) {
		if (!activePanel) return;
		// .panel 자체가 화면 전체를 덮는 오버레이다. 따라서 "바깥"의 기준은
		// .panel 이 아니라 실제 내용 상자인 .panel__inner 여야 한다.
		var inner = activePanel.querySelector('.panel__inner');
		if (inner ? inner.contains(e.target) : activePanel.contains(e.target)) return;
		if (card && card.contains(e.target)) return; // 헤더 영역 클릭 방어
		goHome();
	}

	function openPanel(id, opts) {
		opts = opts || {};
		var panel = panelMap[id];
		if (!panel || panel === activePanel) return;

		if (activePanel) {
			// 패널→패널 직접 전환은 중간에 홈 상태를 거치지 않으므로 즉시 정리한다
			activePanel.classList.remove('is-active');
			activePanel.hidden = true;
		}

		panel.hidden = false;
		body.classList.add('is-panel-open');
		activePanel = panel;
		if (opts.trigger) lastTrigger = opts.trigger;

		if (reduceMotion || opts.immediate) {
			panel.classList.add('is-active');
		} else {
			// hidden 해제와 is-active 부여 사이에 프레임 경계를 둬서 전환이 실제로 걸리게 한다
			requestAnimationFrame(function () {
				requestAnimationFrame(function () {
					panel.classList.add('is-active');
				});
			});
		}

		var title = panel.querySelector('.panel__title');
		if (title) {
			title.setAttribute('tabindex', '-1');
			title.focus();
		}

		document.addEventListener('keydown', onKeydown);
		// 이 클릭이 방금 패널을 연 그 클릭이면 즉시 다시 닫히므로, 한 틱 미뤄서 등록한다
		setTimeout(function () {
			document.addEventListener('click', onOutsideClick);
		}, 0);
	}

	function closePanel() {
		if (!activePanel) return;
		var panel = activePanel;
		activePanel = null;
		body.classList.remove('is-panel-open');
		panel.classList.remove('is-active');

		document.removeEventListener('keydown', onKeydown);
		document.removeEventListener('click', onOutsideClick);

		if (reduceMotion) {
			panel.hidden = true;
		} else {
			var done = false;
			var finish = function () {
				if (done) return;
				done = true;
				panel.removeEventListener('transitionend', finish);
				panel.hidden = true;
			};
			panel.addEventListener('transitionend', finish);
			setTimeout(finish, TRANSITION_FALLBACK);
		}

		var toFocus = lastTrigger || navLinks[0];
		if (toFocus) toFocus.focus();
		lastTrigger = null;
	}

	function goHome() {
		if (!activePanel) return;
		history.pushState(null, '', location.pathname + location.search);
		closePanel();
	}

	// 내비 링크 클릭 → 해시 갱신 + 패널 열기
	navLinks.forEach(function (link) {
		link.addEventListener('click', function (e) {
			var id = (link.getAttribute('href') || '').slice(1);
			if (!panelMap[id]) return; // 매핑 안 되는 링크는 기본 동작에 맡긴다
			e.preventDefault();
			history.pushState(null, '', '#' + id);
			openPanel(id, { trigger: link });
			closeMobileMenu();
		});
	});

	// 패널 내부의 닫기 버튼(이벤트 위임)
	panelsRoot.addEventListener('click', function (e) {
		if (e.target.closest && e.target.closest('.panel__close')) {
			goHome();
		}
	});

	// 뒤로/앞으로 가기
	window.addEventListener('popstate', function () {
		var id = panelIdFromHash();
		if (id) {
			openPanel(id, { trigger: linkForId(id) });
		} else {
			closePanel();
		}
	});

	// 모바일 메뉴 토글 — 마크업에 없으면 조용히 건너뛴다
	var menuBtn = document.querySelector('.card__menu-btn');
	var navList = document.querySelector('.navlist');
	function closeMobileMenu() {
		if (!menuBtn || !navList) return;
		navList.classList.remove('is-open');
		menuBtn.setAttribute('aria-expanded', 'false');
	}
	if (menuBtn && navList) {
		menuBtn.addEventListener('click', function () {
			var open = navList.classList.toggle('is-open');
			menuBtn.setAttribute('aria-expanded', open ? 'true' : 'false');
		});
	}

	// 문의 폼 — 실제 전송 엔드포인트가 없으므로 접수됐다고 말하지 않는다
	var form = document.getElementById('contactForm');
	var status = document.getElementById('formStatus');
	if (form && status) {
		form.addEventListener('submit', function (e) {
			e.preventDefault();
			var name = form.elements.name ? form.elements.name.value.trim() : '';
			var email = form.elements.email ? form.elements.email.value.trim() : '';

			var show = function (message, ok) {
				status.textContent = message;
				status.hidden = false;
				status.classList.toggle('is-ok', ok);
				status.classList.toggle('is-error', !ok);
			};

			if (!name || !email) {
				show('이름과 이메일을 입력해 주세요.', false);
				return;
			}
			if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
				show('올바른 이메일 주소를 입력해 주세요.', false);
				return;
			}
			show('입력값은 확인했지만, 이 페이지는 실제로 메시지를 전송하지 않습니다. 다른 채널로 연락해 주세요.', true);
		});
	}

	// 최초 진입: 해시가 유효한 패널이면 전환 없이 즉시 연다. 무효한 해시는 홈 상태 유지.
	var initialId = panelIdFromHash();
	if (initialId) {
		openPanel(initialId, { immediate: true, trigger: linkForId(initialId) });
	}
})();
