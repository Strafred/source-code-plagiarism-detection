public class Main {
    static java.util.Scanner sc = new java.util.Scanner(System.in);
    public static int input(){
        System.out.print("Enter an integer: ");
        return sc.nextInt();
    }
    public static void main(String[] args) {
        int nmb=input();
        reverse(nmb);
    }
    public static void reverse(int nmb) {
        int get=0;
        while (nmb != 0) {
            get = nmb % 10;
            System.out.print(get);
            nmb = nmb / 10;
        }
        System.out.println();
    }
}
