/* ── EcoAlert — Authentication & Session ─────────────────────────────────── */

function authApp() {
  return {
    /* ── State ─────────────────────────────────────────────────────────── */
    user: null,
    token: localStorage.getItem('eco_token'),
    isLoggedIn: false,
    
    // Auth modals
    showAuthModal: false,
    authMode: 'login', // 'login' | 'register'
    
    // Forms
    loginData: { username: '', password: '' },
    registerData: { username: '', email: '', password: '' },
    authLoading: false,

    /* ── Init ─────────────────────────────────────────────────────────── */
    async initAuth() {
      if (this.token) {
        await this.fetchMe();
      }
    },

    /* ── Actions ───────────────────────────────────────────────────────── */
    async login() {
      this.authLoading = true;
      try {
        const formData = new URLSearchParams();
        formData.append('username', this.loginData.username);
        formData.append('password', this.loginData.password);

        const data = await fetchAPI('/api/auth/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
          body: formData
        });

        this.token = data.access_token;
        localStorage.setItem('eco_token', this.token);
        await this.fetchMe();
        this.showAuthModal = false;
        showToast(`Bine ai revenit, ${this.user.username}!`);
        
        // Refresh other modules if needed
        window.dispatchEvent(new CustomEvent('eco:authChanged'));
      } catch (e) {
        showToast(e.message, 'error');
      } finally {
        this.authLoading = false;
      }
    },

    async register() {
      this.authLoading = true;
      try {
        await fetchAPI('/api/auth/register', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(this.registerData)
        });

        showToast('Cont creat cu succes! Te poți loga acum. [MOC-MAIL: Verifică consola]');
        this.authMode = 'login';
        this.loginData.username = this.registerData.username;
      } catch (e) {
        showToast(e.message, 'error');
      } finally {
        this.authLoading = false;
      }
    },

    async fetchMe() {
      try {
        this.user = await fetchAPI('/api/auth/me');
        this.isLoggedIn = true;
      } catch (e) {
        this.logout();
      }
    },

    logout() {
      this.token = null;
      this.user = null;
      this.isLoggedIn = false;
      localStorage.removeItem('eco_token');
      showToast('Te-ai delogat cu succes.');
      window.dispatchEvent(new CustomEvent('eco:authChanged'));
    },

    openAuth(mode = 'login') {
      this.authMode = mode;
      this.showAuthModal = true;
    }
  };
}
