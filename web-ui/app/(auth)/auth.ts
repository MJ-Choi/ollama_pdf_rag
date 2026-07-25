import { compare } from "bcrypt-ts";
import NextAuth, { type DefaultSession } from "next-auth";
import type { DefaultJWT } from "next-auth/jwt";
import Credentials from "next-auth/providers/credentials";
import { cookies } from "next/headers";
import { DUMMY_PASSWORD } from "@/lib/constants";
import { createGuestUser, getUser, getUserById } from "@/lib/db/queries";
import { authConfig } from "./auth.config";

// Separate from the NextAuth session cookie so guest identity survives a
// session-cookie rotation (JWT expiry, browser cookie clearing on the auth
// cookie specifically, etc.) — without this, every re-login mints a brand
// new guest `user` row, and any chat created under the old guest id becomes
// permanently un-deletable (chat.userId can never match the new session's
// user.id again). ~400 days is the practical browser-enforced cookie cap.
const GUEST_ID_COOKIE = "guest-user-id";
const GUEST_ID_COOKIE_MAX_AGE = 60 * 60 * 24 * 400;

export type UserType = "guest" | "regular";

declare module "next-auth" {
  interface Session extends DefaultSession {
    user: {
      id: string;
      type: UserType;
    } & DefaultSession["user"];
  }

  // biome-ignore lint/nursery/useConsistentTypeDefinitions: "Required"
  interface User {
    id?: string;
    email?: string | null;
    type: UserType;
  }
}

declare module "next-auth/jwt" {
  interface JWT extends DefaultJWT {
    id: string;
    type: UserType;
  }
}

export const {
  handlers: { GET, POST },
  auth,
  signIn,
  signOut,
} = NextAuth({
  ...authConfig,
  providers: [
    Credentials({
      credentials: {},
      async authorize({ email, password }: any) {
        const users = await getUser(email);

        if (users.length === 0) {
          await compare(password, DUMMY_PASSWORD);
          return null;
        }

        const [user] = users;

        if (!user.password) {
          await compare(password, DUMMY_PASSWORD);
          return null;
        }

        const passwordsMatch = await compare(password, user.password);

        if (!passwordsMatch) {
          return null;
        }

        return { ...user, type: "regular" };
      },
    }),
    Credentials({
      id: "guest",
      credentials: {},
      async authorize() {
        const cookieStore = await cookies();
        const existingGuestId = cookieStore.get(GUEST_ID_COOKIE)?.value;

        if (existingGuestId) {
          const [existingUser] = await getUserById(existingGuestId);
          if (existingUser) {
            return { ...existingUser, type: "guest" };
          }
          console.log(
            `Guest cookie pointed at missing user ${existingGuestId}, creating a new guest identity`
          );
        }

        const [guestUser] = await createGuestUser();
        cookieStore.set(GUEST_ID_COOKIE, guestUser.id, {
          httpOnly: true,
          maxAge: GUEST_ID_COOKIE_MAX_AGE,
          sameSite: "lax",
        });
        return { ...guestUser, type: "guest" };
      },
    }),
  ],
  callbacks: {
    jwt({ token, user }) {
      if (user) {
        token.id = user.id as string;
        token.type = user.type;
      }

      return token;
    },
    session({ session, token }) {
      if (session.user) {
        session.user.id = token.id;
        session.user.type = token.type;
      }

      return session;
    },
  },
});
