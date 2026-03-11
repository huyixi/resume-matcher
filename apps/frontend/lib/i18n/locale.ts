import { defaultLocale, locales, type Locale } from '@/i18n/config';

const htmlLangByLocale: Record<Locale, string> = {
  en: 'en-US',
  es: 'es-ES',
  zh: 'zh-CN',
  ja: 'ja-JP',
  pt: 'pt-BR',
};

function normalizeLocale(value: string | undefined): string | null {
  if (!value) return null;
  return value.trim().replace(/_/g, '-').toLowerCase();
}

export function resolveLocale(value: string | undefined): Locale {
  const normalized = normalizeLocale(value);
  if (!normalized) {
    return defaultLocale;
  }

  if (locales.includes(normalized as Locale)) {
    return normalized as Locale;
  }

  const [languageCode] = normalized.split('-');
  if (languageCode && locales.includes(languageCode as Locale)) {
    return languageCode as Locale;
  }

  return defaultLocale;
}

export function detectPreferredLocale(languages: readonly string[] | undefined): Locale {
  if (!languages?.length) {
    return defaultLocale;
  }

  for (const language of languages) {
    const normalized = normalizeLocale(language);
    if (!normalized) {
      continue;
    }

    if (locales.includes(normalized as Locale)) {
      return normalized as Locale;
    }

    const [languageCode] = normalized.split('-');
    if (languageCode && locales.includes(languageCode as Locale)) {
      return languageCode as Locale;
    }
  }

  return defaultLocale;
}

export function getHtmlLang(locale: Locale): string {
  return htmlLangByLocale[locale];
}
