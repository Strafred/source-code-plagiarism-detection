public class Main {
    public static void main(String[] args) {
        System.out.println("Miles\t\tKilometers");
        System.out.println("-------------------------------");
        int mile = 1;
        while (mile <= 10) {
                System.out.println(mile + "\t\t" + mile * 1.609);
                mile++;
        }
    }
}
