public class Main {
    private static int mile;
    public static void main(String[] args) {
        mile = 1;
        System.out.println("Miles\t\tKilometers");
        System.out.println("-------------------------------");
        loops();
    }
    public static void loops(){
        for (int i = 10; i > 0; i--) {
            System.out.println(mile + "\t\t" + mile * 1.609);
                mile++;
        }
    }
}
