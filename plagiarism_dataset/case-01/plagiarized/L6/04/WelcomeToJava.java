public class WelcomeToJava {
    public static String test(int i) {
        if(i>0) {
            System.out.println("Welcome To Java");
            return test(--i);
        }
        else
            return "";
    }
    public static void main(String[] args) {
        int i=5;
        test(i);
    }
}
